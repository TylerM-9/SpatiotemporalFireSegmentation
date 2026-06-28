# train_voc_msra_unet.py
from __future__ import absolute_import, division, print_function

import os
from datetime import datetime
import socket
import timeit
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import imageio
import matplotlib.pyplot as plt

from torchvision.transforms import InterpolationMode

# --- your paths, datasets, and transforms ---
from dataloaders import joint_transforms, voc_msra_dataloader as db
from mypath import Path

# --- your model: attention-ready U-Net (or plain U-Net) ---
from network.UNet_new import UNet

# -------------------
# Config
# -------------------
class TrainingConfig:
    def __init__(self):
        self.gpu_id: int   = 0
        self.last_iter: int = 0
        self.iter_num: int  = 20000        # total optimization steps (not epochs)
        self.batch_size: int = 8
        self.snapshot: int   = 5000         # save checkpoint every N iters
        self.lr: float       = 1e-3
        self.wd: float       = 5e-4
        self.lr_decay: float = 0.9          # poly power
        self.model_name: str = 'Seg_Branch_UNet_VOC_MSRA'

# -------------------
# Trainer
# -------------------
class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.save_dirs = self._setup_directories()
        self.writer = self._setup_tensorboard()
        self.loss_values: List[float] = []

    # ---- setup ----
    def _setup_device(self) -> torch.device:
        device = torch.device(f"cuda:{self.config.gpu_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.gpu_id)
            print(f'Using GPU: {self.config.gpu_id}')
        return device

    def _setup_directories(self) -> Dict[str, str]:
        save_root = Path.save_root_dir()
        model_dir = os.path.join(save_root, self.config.model_name)
        os.makedirs(save_root, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        return {'save_dir': save_root, 'model_dir': model_dir}

    def _setup_tensorboard(self) -> SummaryWriter:
        log_dir = os.path.join(
            self.save_dirs['save_dir'],
            'SegBranch_runs',
            f'{datetime.now().strftime("%b%d_%H-%M-%S")}_{socket.gethostname()}'
        )
        return SummaryWriter(log_dir=log_dir, comment='-voc-msra-unet')

    # ---- data & model ----
    def _get_transforms(self):
        joint_transform = joint_transforms.Compose([
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomRotate(10)
        ])

        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        target_transform = transforms.ToTensor()  # produces 0..1
        return joint_transform, img_transform, target_transform

    def _initialize_network(self) -> nn.Module:
        # Model returns logits (no final activation) for BCEWithLogitsLoss
        net = UNet(
            in_ch=3,
            base=64,
            out_ch=1,
            bilinear=True,
            final_act="none",   # <— logits
            use_aux=False       # <— single output tensor
        )
        return net

    def _setup_optimizer(self, net: nn.Module):
        # Simple SGD (as in your examples); poly LR adjusted per-iteration below
        return optim.SGD(
            net.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd
        )

    # ---- training helpers ----
    @staticmethod
    def _poly_lr(base_lr: float, progress: float, power: float) -> float:
        # progress in [0,1]
        return base_lr * (1.0 - progress) ** power

    def _save_checkpoint(self, net: nn.Module, curr_iter: int):
        ckpt_path = os.path.join(self.save_dirs['model_dir'], f'iter_{curr_iter}.pth')
        torch.save({'iter': curr_iter, 'state_dict': net.state_dict()}, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    def _save_visualization(self, inputs: torch.Tensor, gts: torch.Tensor, pred: torch.Tensor, curr_iter: int):
        # inputs: [B,3,H,W] (normalized), gts: [B,1,H,W] (0/1), pred: [B,1,H,W] logits
        with torch.no_grad():
            # de-normalize first item for visualization
            x = inputs[0].detach().cpu().numpy().transpose(1, 2, 0)
            x = (x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8)

            # ground truth to 3-ch
            gt = gts[0].detach().cpu().numpy()  # [1,H,W] or [H,W]
            if gt.ndim == 3:
                gt = gt[0]
            gt = (gt * 255).astype(np.uint8)
            gt = np.repeat(gt[..., None], 3, axis=2)

            # prediction: logits -> prob (sigmoid)
            pr = torch.sigmoid(pred[0, 0]).detach().cpu().numpy()
            pr_vis = (pr * 255).astype(np.uint8)
            pr_vis = np.repeat(pr_vis[..., None], 3, axis=2)

            canvas = np.concatenate([pr_vis, gt, x], axis=0)
            res_dir = os.path.join(self.save_dirs['save_dir'], f"{self.config.model_name}_results")
            os.makedirs(res_dir, exist_ok=True)
            imageio.imwrite(os.path.join(res_dir, f"iter_{curr_iter}.png"), canvas)

    def _plot_and_save_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_values, label='Training Loss')
        plt.xlabel('Iterations (logged every 10 iters)')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.save_dirs['model_dir'], 'loss_curve.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss curve saved to {plot_path}")

    # ---- main train ----
    def train(self):
        joint_transform, img_transform, target_transform = self._get_transforms()

        # Dataset: expects voc_msra_dataloadr_256(root_msra, root_voc, joint, img, target)
        train_set = db.voc_msra_dataloadr_256(
            Path.MSRAdataset_dir(),
            Path.VOC_dir(),
            joint_transform,
            img_transform,
            target_transform
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True
        )

        net = self._initialize_network().to(self.device)
        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = self._setup_optimizer(net)

        curr_iter = self.config.last_iter
        total_iters = self.config.iter_num
        start_time = timeit.default_timer()

        net.train()
        while curr_iter < total_iters:
            for sample_batched in train_loader:
                # learning-rate schedule (poly)
                progress = min(1.0, curr_iter / max(1, total_iters))
                lr_now = self._poly_lr(self.config.lr, progress, self.config.lr_decay)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_now

                # fetch & send to device
                inputs, gts = sample_batched['images'], sample_batched['gts']
                inputs = inputs.to(self.device, non_blocking=True)
                gts    = gts.to(self.device, non_blocking=True)

                # ensure masks are [B,1,H,W] float in {0,1}
                if gts.dim() == 3:
                    gts = gts.unsqueeze(1)
                gts = gts.float()

                optimizer.zero_grad(set_to_none=True)
                pred = net(inputs)         # logits: [B,1,H,W]
                loss = criterion(pred, gts)
                loss.backward()
                optimizer.step()

                # logging
                if curr_iter % 10 == 0:
                    self.loss_values.append(loss.item())
                    elapsed = timeit.default_timer() - start_time
                    print(f"Iter: {curr_iter:6d}/{total_iters:6d} | "
                          f"Loss: {loss.item():.6f} | LR: {lr_now:.6e} | "
                          f"Time(s): {elapsed:7.2f}")
                    self.writer.add_scalar('loss/train', loss.item(), curr_iter)
                    self.writer.add_scalar('lr', lr_now, curr_iter)

                # quick vis every 1000 iters
                if curr_iter % 1000 == 0 and curr_iter > 0:
                    self._save_visualization(inputs, gts, pred, curr_iter)

                # checkpoint
                if curr_iter % self.config.snapshot == 0 and curr_iter > 0:
                    self._save_checkpoint(net, curr_iter)

                curr_iter += 1
                if curr_iter >= total_iters:
                    break

        # final checkpoint + loss curve
        self._save_checkpoint(net, curr_iter)
        self._plot_and_save_loss()
        self.writer.close()
        print("Training completed!")

# -------------------
# Entrypoint
# -------------------
def main():
    cfg = TrainingConfig()
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
