from __future__ import absolute_import, division, print_function

import os
from datetime import datetime
import socket
import timeit
from typing import Dict, List
import sys

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

# Add Swin-UNET path - CHANGE THIS TO YOUR PATH
from network.Swin_Unet.networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


from dataloaders import FIRE_dataloader as db
from mypath import Path
from dataloaders import custom_transforms as tr

gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")


class TrainingConfig:
    def __init__(self):
        self.gpu_id = 0
        self.last_epoch = 0
        self.num_epochs = 201
        self.batch_size = 8
        self.snapshot_freq = 10
        self.lr = 0.05  # Changed for Swin-UNET
        self.wd = 1e-4  # Changed for Swin-UNET
        self.lr_decay = 0.9
        self.lr_decay_freq = 50
        self.side_weight = 0.5
        self.model_name = 'Seg_Branch_SwinUNET_flame_spatio_temporal'
        self.validation_freq = 10


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.save_dirs = self._setup_directories()
        self.writer = self._setup_tensorboard()
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_pas = []
        self.epochs = []

    def _setup_device(self) -> torch.device:
        device = torch.device(f"cuda:{self.config.gpu_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f'Using GPU: {self.config.gpu_id}')
        else:
            print('Using CPU')
        return device

    def _setup_directories(self) -> Dict[str, str]:
        dirs = {
            'save_dir': Path.save_root_dir(),
            'model_dir': os.path.join(Path.save_root_dir(), self.config.model_name),
            'results_dir': os.path.join(Path.save_root_dir(), f"{self.config.model_name}_results")
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
        # Training transforms (with augmentation)
        train_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
        ])

        # Validation transforms (no augmentation)
        val_transforms = None

        return train_transforms, val_transforms

    def _initialize_network(self):
        # Create Swin-UNET model
        net = SwinTransformerSys(
            img_size=256,
            patch_size=4,
            in_chans=3,
            num_classes=1,  # Binary segmentation - change if needed
            embed_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,  # 8 for 256x256 images
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )

        # Load pretrained weights if available
        pretrained_path = "$HOME/STCNN/network/swin_tiny_patch4_window7_224.pth"
        if os.path.exists(pretrained_path):
            print("Loading pretrained Swin-Transformer weights")
            try:
                pretrained_dict = torch.load(pretrained_path, map_location=self.device)

                # Handle different checkpoint formats
                if 'model' in pretrained_dict:
                    pretrained_dict = pretrained_dict['model']
                elif 'state_dict' in pretrained_dict:
                    pretrained_dict = pretrained_dict['state_dict']

                model_dict = net.state_dict()

                # Filter out unnecessary keys and handle shape mismatches
                filtered_dict = {}
                missing_keys = []
                shape_mismatches = []

                for k, v in pretrained_dict.items():
                    k_clean = k.replace('module.', '')
                    if k_clean in model_dict:
                        if v.shape == model_dict[k_clean].shape:
                            filtered_dict[k_clean] = v
                        else:
                            shape_mismatches.append(k_clean)
                            print(f"Shape mismatch for {k_clean}: pretrained {v.shape} vs model {model_dict[k_clean].shape}")
                    else:
                        missing_keys.append(k_clean)

                print(f"Loading {len(filtered_dict)} out of {len(model_dict)} parameters")
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if shape_mismatches:
                    print(f"Shape mismatches: {len(shape_mismatches)}")

                model_dict.update(filtered_dict)
                net.load_state_dict(model_dict, strict=False)
                print("Successfully loaded pretrained weights")

            except Exception as e:
                print(f"Failed to load pretrained weights: {e}")
                print("Training from scratch")
        else:
            print("No pretrained weights found. Training from scratch")

        return net

    def _setup_optimizer(self, net):
        return optim.SGD(net.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)

    def _setup_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_freq, gamma=self.config.lr_decay)

    def calculate_metrics(self, predictions, targets, threshold=0.5):
        """Calculate IoU and Pixel Accuracy"""
        # Apply sigmoid and threshold
        pred_probs = torch.sigmoid(predictions)
        pred_binary = (pred_probs > threshold).float()
        target_binary = (targets > threshold).float()

        # Calculate IoU
        intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
        union = (pred_binary + target_binary).clamp(0, 1).sum(dim=(1, 2, 3))
        iou = (intersection / (union + 1e-6)).mean().item()

        # Calculate Pixel Accuracy
        correct = (pred_binary == target_binary).sum(dim=(1, 2, 3))
        total = torch.numel(target_binary[0])
        pa = (correct / total).mean().item()

        return iou, pa

    def validate(self, val_loader, net, criterion):
        """Validation loop"""
        net.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_pa = 0.0
        num_batches = 0

        with torch.no_grad():
            for sample_batched in val_loader:
                inputs, gts = sample_batched['images'], sample_batched['seg_gt']
                inputs, gts = inputs.to(self.device), gts.to(self.device).float()

                pred = net(inputs)
                loss = criterion(pred, gts)

                # Calculate metrics
                iou, pa = self.calculate_metrics(pred, gts)

                val_loss += loss.item()
                val_iou += iou
                val_pa += pa
                num_batches += 1

        return val_loss / num_batches, val_iou / num_batches, val_pa / num_batches

    def save_visualization(self, inputs, gts, pred, epoch: int, phase: str = "train"):
        """Save visualization images"""
        # Convert input image back to original scale
        inputs_np = inputs[0].cpu().numpy().transpose(1, 2, 0)
        if inputs_np.min() < 0:  # If normalized
            inputs_np = (inputs_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        inputs_np = np.clip(inputs_np * 255, 0, 255).astype(np.uint8)

        # Convert ground truth to 3-channel image
        gt_np = (gts[0].cpu().numpy().squeeze() * 255).astype(np.uint8)
        gt_np = np.stack([gt_np, gt_np, gt_np], axis=-1)

        # Convert prediction to 3-channel image
        pred_np = (torch.sigmoid(pred[0]).cpu().detach().numpy().squeeze() > 0.5) * 255
        pred_np = pred_np.astype(np.uint8)
        pred_np = np.stack([pred_np, pred_np, pred_np], axis=-1)

        # Concatenate images
        samples = np.concatenate((pred_np, gt_np, inputs_np), axis=0)
        samples = np.clip(samples, 0, 255).astype(np.uint8)

        # Save image
        filename = f"{phase}_epoch_{epoch:04d}.png"
        imageio.imwrite(os.path.join(self.save_dirs['results_dir'], filename), samples)

    def save_checkpoint(self, net, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'val_pas': self.val_pas
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dirs['model_dir'], f'epoch_{epoch:04d}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dirs['model_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def load_checkpoint(self, net, optimizer, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_ious = checkpoint.get('val_ious', [])
        self.val_pas = checkpoint.get('val_pas', [])

        return checkpoint['epoch']

    def plot_training_curves(self):
        """Plot and save training curves"""
        if not self.epochs:
            print("No training data to plot")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Training and Validation Loss (combined on one graph)
        ax.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if self.val_losses:
            val_epochs = self.epochs[::self.config.validation_freq][:len(self.val_losses)]
            ax.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)

        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.save_dirs['model_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {plot_path}")
        plt.close()

    def train(self):
        # Setup data loaders
        train_transforms, val_transforms = self._get_transforms()

        # Training dataset
        db_train = db.FIREDatasetRandom(inputRes=(256, 256), transform=train_transforms, mode="train", num_frame=1)
        trainloader = DataLoader(db_train, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

        # Validation dataset
        db_val = db.FIREDatasetRandom(inputRes=(256, 256), transform=val_transforms, mode="test", num_frame=1)
        val_loader = DataLoader(db_val, batch_size=self.config.batch_size, num_workers=4, shuffle=False)

        print(f"Training samples: {len(db_train)}")
        print(f"Validation samples: {len(db_val)}")

        # Initialize network, criterion, optimizer, scheduler
        net = self._initialize_network().to(self.device)
        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = self._setup_optimizer(net)
        scheduler = self._setup_scheduler(optimizer)

        # Load checkpoint if resuming training
        start_epoch = self.config.last_epoch
        if start_epoch > 0:
            checkpoint_path = os.path.join(self.save_dirs['model_dir'], f'epoch_{start_epoch:04d}.pth')
            if os.path.exists(checkpoint_path):
                start_epoch = self.load_checkpoint(net, optimizer, checkpoint_path)
                print(f"Resuming training from epoch {start_epoch}")

        best_val_iou = 0.0

        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 50)

            # Training phase
            net.train()
            epoch_train_loss = 0.0
            num_batches = 0

            start_time = timeit.default_timer()

            for batch_idx, sample_batched in enumerate(trainloader):
                inputs, gts = sample_batched['images'], sample_batched['seg_gt']
                inputs, gts = inputs.to(self.device), gts.to(self.device).float()

                optimizer.zero_grad()
                pred = net(inputs)
                loss = criterion(pred, gts)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                num_batches += 1

                # Print progress
                if batch_idx % 50 == 0:
                    print(f'Batch {batch_idx}/{len(trainloader)}, Loss: {loss.item():.4f}')

            avg_train_loss = epoch_train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            self.epochs.append(epoch + 1)

            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f'Training Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}')

            # Validation phase
            if (epoch + 1) % self.config.validation_freq == 0:
                print("Validating...")
                val_loss, val_iou, val_pa = self.validate(val_loader, net, criterion)

                self.val_losses.append(val_loss)
                self.val_ious.append(val_iou)
                self.val_pas.append(val_pa)

                print(f'Validation - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, PA: {val_pa:.4f}')

                # Save visualization
                net.eval()
                with torch.no_grad():
                    sample = next(iter(val_loader))
                    inputs, gts = sample['images'], sample['seg_gt']
                    inputs, gts = inputs.to(self.device), gts.to(self.device).float()
                    pred = net(inputs)
                    self.save_visualization(inputs, gts, pred, epoch + 1, "val")

                # Check if best model
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    self.save_checkpoint(net, optimizer, epoch + 1, is_best=True)
                    print(f'New best IoU: {best_val_iou:.4f}')

            # Save checkpoint every snapshot_freq epochs
            if (epoch + 1) % self.config.snapshot_freq == 0:
                self.save_checkpoint(net, optimizer, epoch + 1)

            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch + 1)

            if self.val_losses:
                self.writer.add_scalar('Loss/Validation', self.val_losses[-1], epoch + 1)
                self.writer.add_scalar('IoU/Validation', self.val_ious[-1], epoch + 1)
                self.writer.add_scalar('PixelAccuracy/Validation', self.val_pas[-1], epoch + 1)

            epoch_time = timeit.default_timer() - start_time
            print(f'Epoch time: {epoch_time:.2f}s')

        # Final checkpoint and plots
        self.save_checkpoint(net, optimizer, self.config.num_epochs)
        self.plot_training_curves()
        self.writer.close()

        print("\nTraining completed!")
        print(f"Best validation IoU: {best_val_iou:.4f}")


def main():
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()