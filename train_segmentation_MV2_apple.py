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
from torchvision import transforms, models
from tensorboardX import SummaryWriter
import imageio
import matplotlib.pyplot as plt

from network.joint_pred_seg import SegBranch, SegEncoder, SegDecoder
from network.MV2_try import SegEncoder_MobileViT2_Compat
from dataloaders import joint_transforms, voc_msra_dataloader as db
from mypath import Path

class TrainingConfig:
    def __init__(self):
        self.gpu_id = 0
        self.last_iter = 0
        self.iter_num = 12000
        self.batch_size = 8
        self.snapshot = 1000
        self.lr = 1e-3
        self.wd = 5e-4
        self.lr_decay = 0.9
        self.side_weight = 0.5
        self.model_name = 'Seg_Branch_CBAM'

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.save_dirs = self._setup_directories()
        self.writer = self._setup_tensorboard()
        self.loss_values = []
        
    def _setup_device(self) -> torch.device:
        device = torch.device(f"cuda:{self.config.gpu_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f'Using GPU: {self.config.gpu_id}')
        return device

    def _setup_directories(self) -> Dict[str, str]:
        dirs = {
            'save_dir': Path.save_root_dir(),
            'model_dir': os.path.join(Path.save_root_dir(), self.config.model_name)
        }
        
        for directory in dirs.values():
            os.makedirs(directory, exist_ok=True)
        
        return dirs

    def _setup_tensorboard(self) -> SummaryWriter:
        log_dir = os.path.join(
            self.save_dirs['save_dir'], 
            'SegBranch_runs', 
            f'{datetime.now().strftime("%b%d_%H-%M-%S")}_{socket.gethostname()}'
        )
        return SummaryWriter(log_dir=log_dir, comment='-parent')

    def _get_transforms(self):
        joint_transform = joint_transforms.Compose([
            joint_transforms.RandomCrop(300),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomRotate(10)
        ])
        
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        target_transform = transforms.ToTensor()
        
        return joint_transform, img_transform, target_transform

    def _initialize_network(self):
        # Swap in MobileViT-v2 encoder; keep your decoder exactly as-is
        encoder = SegEncoder_MobileViT2_Compat(
            mv2_variant="mobilevitv2_100",   # or _075, _150, etc.
            pretrained=True,                 # ImageNet-pretrained via timm
            out_indices=(1, 2, 3, 4),        # 4 scales (≈ strides 4/8/16/32)
            target_planes=(256, 512, 1024, 2048)  # match SegDecoderCBAM's expectations
        )
        decoder = SegDecoder()
        return SegBranch(net_enc=encoder, net_dec=decoder)

    def _initialize_encoder(self, net: SegEncoder):
        print("Loading weights from PyTorch ResNet101")
        resnet = models.resnet101(pretrained=True)
        model_dict = net.state_dict()
        model_dict.update(resnet.state_dict())
        net.load_state_dict(model_dict, strict=False)

    def save_visualization(self, inputs, gts, pred, curr_iter: int):
        inputs_np = self._prepare_input_visualization(inputs[0])
        gt_np = self._prepare_gt_visualization(gts[0])
        pred_np = self._prepare_pred_visualization(pred[-1][0])
        
        samples = np.concatenate((pred_np, gt_np, inputs_np), axis=0)
        samples = np.clip(samples, 0, 255).astype(np.uint8)
        
        running_res_dir = os.path.join(self.save_dirs['save_dir'], f"{self.config.model_name}_results")
        os.makedirs(running_res_dir, exist_ok=True)
        imageio.imwrite(os.path.join(running_res_dir, f"train_{curr_iter}.png"), samples)

    def train(self):
        joint_transform, img_transform, target_transform = self._get_transforms()
        train_set = db.voc_msra_dataloadr(
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
            shuffle=True
        )

        net = self._initialize_network().to(self.device)
        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = self._setup_optimizer(net)

        if self.config.last_iter > 0:
            self._load_checkpoint(net)

        curr_iter = self.config.last_iter
        
        while curr_iter < self.config.iter_num:
            curr_iter = self._train_epoch(
                train_loader, 
                net, 
                criterion, 
                optimizer, 
                curr_iter
            )

        self._plot_loss_curve()

    def _train_epoch(self, train_loader, net, criterion, optimizer, curr_iter):
        start_time = timeit.default_timer()
        
        for sample_batched in train_loader:
            curr_iter = self._train_iteration(
                sample_batched, 
                net, 
                criterion, 
                optimizer, 
                curr_iter, 
                start_time
            )
            
            if curr_iter >= self.config.iter_num:
                break
                
        return curr_iter

    # Additional helper methods would go here...

def main():
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
