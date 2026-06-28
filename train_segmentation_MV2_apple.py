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
from torchvision.transforms import InterpolationMode

from network.UNet_models import ResUNet
from dataloaders import joint_transforms, voc_msra_dataloader as db
from mypath import Path

class TrainingConfig:
    def __init__(self):
        self.gpu_id = 0
        self.last_iter = 0
        self.iter_num = 3
        self.batch_size = 8
        self.snapshot = 5000
        self.lr = 1e-3
        self.wd = 5e-4
        self.lr_decay = 0.9
        self.side_weight = 0.5
        self.model_name = 'Seg_Branch_ResNet'

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
        return ResUNet(in_channels=3, out_channels=1, init_features=16)

    def save_visualization(self, inputs, gts, pred, curr_iter: int):
        # Convert input image back to original scale
        inputs_np = (inputs[0].cpu().numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        inputs_np = inputs_np.astype(np.uint8)

        # Convert ground truth to binary image
        gt_np = (gts[0].cpu().numpy() * 255).astype(np.uint8)
        gt_np = gt_np.reshape(gt_np.shape[0], gt_np.shape[1], 1)
        gt_np = np.repeat(gt_np, 3, axis=2)
        

        # Convert prediction to binary image
        pred_np = (torch.sigmoid(pred[-1][0]).cpu().detach().numpy() > 0.5) * 255
        pred_np = pred_np.astype(np.uint8)
        pred_np = np.repeat(pred_np[np.newaxis, :, :], 3, axis=0).transpose(1, 2, 0)
        
        samples = np.concatenate((pred_np, gt_np, inputs_np), axis=0)
        samples = np.clip(samples, 0, 255).astype(np.uint8)
        
        running_res_dir = os.path.join(self.save_dirs['save_dir'], f"{self.config.model_name}_results")
        os.makedirs(running_res_dir, exist_ok=True)
        imageio.imwrite(os.path.join(running_res_dir, f"train_{curr_iter}.png"), samples)

    def train(self):
        joint_transform, img_transform, target_transform = self._get_transforms()
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

        # Plot and save the loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_values, label='Training Loss')
        plt.xlabel('Iterations (x10)')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        loss_plot_path = os.path.join(self.save_dirs['model_dir'], 'loss_curve.png')
        plt.savefig(loss_plot_path)
        plt.close()

    def _setup_optimizer(self, net):
        return optim.SGD(net.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)

    def _train_epoch(self, train_loader, net, criterion, optimizer, curr_iter):
        start_time = timeit.default_timer()
        
        for sample_batched in train_loader:
            inputs, gts = sample_batched['images'], sample_batched['gts']
            inputs, gts = inputs.to(self.device), gts.to(self.device)
            
            optimizer.zero_grad()

            pred = net(inputs)  # (B,1,256,256)
            gts = gts.float()  # BCE expects float targets
            loss = criterion(pred, gts)

            loss.backward()
            optimizer.step()
            
            # Log training info
            if curr_iter % 10 == 0:
                print(f'Iter: {curr_iter}, Loss: {loss.item():.4f}')
                self.loss_values.append(loss.item())
                
            # Save checkpoint
            if curr_iter % self.config.snapshot == 0:
                torch.save({
                    'iter': curr_iter,
                    'state_dict': net.state_dict(),
                }, os.path.join(self.save_dirs['model_dir'], f'iter_{curr_iter}.pth'))
                
            curr_iter += 1
            
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
