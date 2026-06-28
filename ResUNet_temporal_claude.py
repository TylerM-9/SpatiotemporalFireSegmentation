from __future__ import absolute_import, division, print_function

import os
from datetime import datetime
import socket
import timeit
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import imageio
import matplotlib.pyplot as plt

# Install: pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp

from dataloaders import FIRE_dataloader as db
from mypath import Path
from dataloaders import custom_transforms as tr

gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")


# ============================================================================
# DECOMPOSED RESUNET ARCHITECTURE
# ============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet encoder extracted from segmentation_models_pytorch.
    Returns multi-scale features for skip connections.
    """

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3):
        super(ResNetEncoder, self).__init__()

        # Create a temporary Unet model to extract the encoder
        temp_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1
        )

        # Extract the encoder
        self.encoder = temp_model.encoder
        self.encoder_name = encoder_name
        self.out_channels = self.encoder.out_channels  # e.g., [3, 64, 64, 128, 256, 512] for resnet34

        print(f"Encoder initialized: {encoder_name}")
        print(f"Encoder output channels: {self.out_channels}")

    def forward(self, x):
        """
        Forward pass through encoder.
        Returns list of features at different scales.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            features: List of [B, C_i, H_i, W_i] tensors at different scales
                     e.g., for resnet34: [x, conv1, layer1, layer2, layer3, layer4]
        """
        features = self.encoder(x)
        return features

    def load_pretrained_weights(self, checkpoint_path):
        """Load encoder weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter encoder weights
        encoder_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder' in k}
        self.encoder.load_state_dict(encoder_dict, strict=False)
        print(f"Loaded encoder weights from {checkpoint_path}")


class ResNetDecoder(nn.Module):
    """
    UNet-style decoder with 3 upsampling blocks and attention-gated skip connections.
    Matches the architecture diagram with proper channel dimensions.
    """

    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64), n_classes=1, use_attention=True):
        """
        Args:
            encoder_channels: List of encoder output channels, e.g., [3, 64, 64, 128, 256, 512]
            decoder_channels: Tuple of 3 decoder channels (256, 128, 64) to match diagram
            n_classes: Number of output classes
            use_attention: Whether to use attention gates (True for "A" blocks in diagram)
        """
        super(ResNetDecoder, self).__init__()

        assert len(decoder_channels) == 3, "Decoder must have exactly 3 blocks to match architecture"

        self.use_attention = use_attention

        # Reverse encoder channels (bottom-up)
        # For ResNet34: [3, 64, 64, 128, 256, 512] -> [512, 256, 128, 64, 64, 3]
        encoder_channels = encoder_channels[::-1]

        # Choose decoder block type
        BlockType = AttentionDecoderBlock if use_attention else DecoderBlock

        # Create 3 decoder blocks matching the diagram
        self.blocks = nn.ModuleList()

        # Block 1: 512 -> 256 (with skip from res4: 256 channels)
        self.blocks.append(
            BlockType(
                in_channels=encoder_channels[0],  # 512 from res5
                out_channels=decoder_channels[0],  # 256
                skip_channels=encoder_channels[1]  # 256 from res4
            )
        )

        # Block 2: 256 -> 128 (with skip from res3: 128 channels)
        self.blocks.append(
            BlockType(
                in_channels=decoder_channels[0],  # 256 from previous block
                out_channels=decoder_channels[1],  # 128
                skip_channels=encoder_channels[2]  # 128 from res3
            )
        )

        # Block 3: 128 -> 64 (with skip from res2: 64 channels)
        self.blocks.append(
            BlockType(
                in_channels=decoder_channels[1],  # 128 from previous block
                out_channels=decoder_channels[2],  # 64
                skip_channels=encoder_channels[3]  # 64 from res2
            )
        )

        # Additional upsampling to reach input resolution
        # After 3 blocks, we're at 1/4 resolution, need 2 more upsamples to reach full res
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[2], decoder_channels[2] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[2] // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[2] // 2, decoder_channels[2] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[2] // 2),
            nn.ReLU(inplace=True),
        )

        # Final segmentation head: 32 -> n_classes
        self.segmentation_head = nn.Conv2d(
            decoder_channels[2] // 2,  # 32 channels after upsampling
            n_classes,
            kernel_size=3,
            padding=1
        )

        attention_str = "WITH attention gates" if use_attention else "WITHOUT attention gates"
        print(f"Decoder initialized with 3 blocks: {decoder_channels} {attention_str}")
        print(
            f"Skip connections from encoder channels: [{encoder_channels[1]}, {encoder_channels[2]}, {encoder_channels[3]}]")

    def forward(self, features):
        """
        Forward pass through decoder with 3 upsampling blocks.

        Args:
            features: List of encoder features [f0, f1, f2, f3, f4, f5]
                     f0 = input (3 channels)
                     f1 = res1 (64 channels)
                     f2 = res2 (64 channels)
                     f3 = res3 (128 channels)
                     f4 = res4 (256 channels)
                     f5 = res5/bottleneck (512 channels)

        Returns:
            x: Segmentation logits [B, n_classes, H, W]
        """
        # Reverse features for bottom-up processing
        features = features[::-1]  # [f5, f4, f3, f2, f1, f0]

        # Block 1: Upsample from res5 (512) and fuse with res4 (256) -> 256
        x = self.blocks[0](features[0], features[1])

        # Block 2: Upsample from 256 and fuse with res3 (128) -> 128
        x = self.blocks[1](x, features[2])

        # Block 3: Upsample from 128 and fuse with res2 (64) -> 64
        x = self.blocks[2](x, features[3])

        # Additional upsampling to reach full input resolution (256x256)
        x = self.final_upsample(x)

        # Final segmentation head: 32 -> n_classes
        x = self.segmentation_head(x)

        return x


class DecoderBlock(nn.Module):
    """
    Single decoder block: Upsample -> Concat Skip -> Conv -> Conv
    """

    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Adjust input channels based on whether we have skip connection
        conv_in_channels = in_channels + skip_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionBlock(nn.Module):
    """
    Attention module for feature fusion (the "A" blocks in the diagram).
    Uses channel and spatial attention to weight skip connections.
    """

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of feature maps in gating signal (from decoder)
            F_l: Number of feature maps in skip connection (from encoder)
            F_int: Number of intermediate feature maps
        """
        super(AttentionBlock, self).__init__()

        # Transform gating signal (from decoder path)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Transform skip connection (from encoder path)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Generate attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder [B, F_g, H, W]
            x: Skip connection from encoder [B, F_l, H, W]

        Returns:
            Attention-weighted skip features [B, F_l, H, W]
        """
        # Transform both inputs to intermediate dimension
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Ensure spatial dimensions match
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # Combine and generate attention map
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B, 1, H, W]

        # Apply attention to skip connection
        return x * psi


class AttentionDecoderBlock(nn.Module):
    """
    Decoder block with attention-gated skip connections.
    Replaces simple concatenation with attention fusion.
    """

    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(AttentionDecoderBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Attention gate for skip connection
        self.attention = None
        if skip_channels > 0:
            self.attention = AttentionBlock(
                F_g=in_channels,  # Gating from upsampled decoder
                F_l=skip_channels,  # Skip from encoder
                F_int=out_channels // 2  # Intermediate dimension
            )

        # Convolutions after fusion
        conv_in_channels = in_channels + skip_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None and self.attention is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x_resized = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            else:
                x_resized = x

            # Apply attention to skip connection
            skip = self.attention(x_resized, skip)

            # Concatenate attended skip with upsampled features
            x = torch.cat([x_resized, skip], dim=1)
        elif skip is not None:
            # Fallback to simple concatenation
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResUNet(nn.Module):
    """
    Complete ResUNet model with decomposed encoder and decoder.
    Can be extended with temporal branches.
    """

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, n_classes=1, decoder_channels=(256, 128, 64, 32, 16)):
        super(ResUNet, self).__init__()

        self.encoder = ResNetEncoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels
        )

        self.decoder = ResNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_classes=n_classes
        )

    def forward(self, x):
        """
        Forward pass through complete model.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            segmentation: Logits [B, n_classes, H, W]
        """
        features = self.encoder(x)
        segmentation = self.decoder(features)
        return segmentation

    def get_encoder_features(self, x):
        """
        Get intermediate encoder features (useful for temporal branch).

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            features: List of multi-scale features
        """
        return self.encoder(x)


# ============================================================================
# TRAINING CONFIGURATION AND TRAINER
# ============================================================================

class TrainingConfig:
    def __init__(self):
        self.gpu_id = 0
        self.last_epoch = 0
        self.num_epochs = 201
        self.batch_size = 8
        self.snapshot_freq = 10
        self.lr = 1e-4  # Lower LR for fine-tuning
        self.wd = 5e-4
        self.lr_decay = 0.9
        self.lr_decay_freq = 50
        self.model_name = 'ResUNet_Decomposed_Attention'
        self.validation_freq = 10

        # ResUNet specific parameters
        self.encoder_name = "resnet34"  # Options: resnet18, resnet34, resnet50, resnet101
        self.encoder_weights = "imagenet"  # Use ImageNet pre-trained weights
        self.in_channels = 3
        self.classes = 1  # Binary segmentation
        self.decoder_channels = (256, 128, 64)  # 3 blocks to match architecture diagram
        self.use_attention = True  # Enable attention gates (the "A" blocks in diagram)


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
            print(f'GPU Name: {torch.cuda.get_device_name(self.config.gpu_id)}')
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
            'ResUNet_runs',
            f'{datetime.now().strftime("%b%d_%H-%M-%S")}_{socket.gethostname()}'
        )
        return SummaryWriter(log_dir=log_dir, comment='-resunet-decomposed')

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
        """
        Initialize decomposed ResUNet
        """
        print(f"Initializing Decomposed ResUNet with {self.config.encoder_name} encoder")
        print(f"Using {self.config.encoder_weights} pre-trained weights")

        # Create decomposed ResUNet model
        net = ResUNet(
            encoder_name=self.config.encoder_name,
            encoder_weights=self.config.encoder_weights,
            in_channels=self.config.in_channels,
            n_classes=self.config.classes,
            decoder_channels=self.config.decoder_channels,
            use_attention=self.config.use_attention
        )

        # Print model information
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in net.encoder.parameters())
        decoder_params = sum(p.numel() for p in net.decoder.parameters())

        print(f"Total parameters: {total_params:,}")
        print(f"Encoder parameters: {encoder_params:,}")
        print(f"Decoder parameters: {decoder_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Optional: Load custom checkpoint if continuing training
        checkpoint_path = os.path.join(self.save_dirs['model_dir'], 'best_model.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net.load_state_dict(checkpoint)
            print("Successfully loaded checkpoint")

        return net

    def _setup_optimizer(self, net):
        """
        Setup optimizer with different learning rates for encoder and decoder
        """
        # Option 1: Same LR for all parameters
        # optimizer = optim.Adam(net.parameters(), lr=self.config.lr, weight_decay=self.config.wd)

        # Option 2: Different LR for encoder (pre-trained) and decoder (random init)
        encoder_params = list(net.encoder.parameters())
        decoder_params = list(net.decoder.parameters())

        optimizer = optim.Adam([
            {'params': encoder_params, 'lr': self.config.lr * 0.1},  # Lower LR for pre-trained
            {'params': decoder_params, 'lr': self.config.lr}
        ], weight_decay=self.config.wd)

        print(f"Optimizer setup: Encoder LR={self.config.lr * 0.1:.6f}, Decoder LR={self.config.lr:.6f}")

        return optimizer

    def _setup_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.lr_decay_freq,
            gamma=self.config.lr_decay
        )

    def calculate_metrics(self, predictions, targets, threshold=0.5):
        """Calculate IoU and Pixel Accuracy"""
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

                iou, pa = self.calculate_metrics(pred, gts)

                val_loss += loss.item()
                val_iou += iou
                val_pa += pa
                num_batches += 1

        return val_loss / num_batches, val_iou / num_batches, val_pa / num_batches

    def save_visualization(self, inputs, gts, pred, epoch: int, phase: str = "train"):
        """Save visualization images"""
        inputs_np = inputs[0].cpu().numpy().transpose(1, 2, 0)
        if inputs_np.min() < 0:
            inputs_np = (inputs_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        inputs_np = np.clip(inputs_np * 255, 0, 255).astype(np.uint8)

        gt_np = (gts[0].cpu().numpy().squeeze() * 255).astype(np.uint8)
        gt_np = np.stack([gt_np, gt_np, gt_np], axis=-1)

        pred_np = (torch.sigmoid(pred[0]).cpu().detach().numpy().squeeze() > 0.5) * 255
        pred_np = pred_np.astype(np.uint8)
        pred_np = np.stack([pred_np, pred_np, pred_np], axis=-1)

        samples = np.concatenate((pred_np, gt_np, inputs_np), axis=0)
        samples = np.clip(samples, 0, 255).astype(np.uint8)

        filename = f"{phase}_epoch_{epoch:04d}.png"
        imageio.imwrite(os.path.join(self.save_dirs['results_dir'], filename), samples)

    def save_checkpoint(self, net, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': {
                'encoder_name': self.config.encoder_name,
                'encoder_weights': self.config.encoder_weights,
                'in_channels': self.config.in_channels,
                'classes': self.config.classes,
                'decoder_channels': self.config.decoder_channels,
            },
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'val_pas': self.val_pas
        }

        checkpoint_path = os.path.join(self.save_dirs['model_dir'], f'epoch_{epoch:04d}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

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

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        if self.val_losses:
            val_epochs = self.epochs[::self.config.validation_freq][:len(self.val_losses)]
            ax2.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss')
            ax2.set_title('Validation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)

        if self.val_ious:
            val_epochs = self.epochs[::self.config.validation_freq][:len(self.val_ious)]
            ax3.plot(val_epochs, self.val_ious, 'g-', label='Validation IoU')
            ax3.set_title('Validation IoU')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('IoU')
            ax3.legend()
            ax3.grid(True)

        if self.val_pas:
            val_epochs = self.epochs[::self.config.validation_freq][:len(self.val_pas)]
            ax4.plot(val_epochs, self.val_pas, 'm-', label='Validation Pixel Accuracy')
            ax4.set_title('Validation Pixel Accuracy')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Pixel Accuracy')
            ax4.legend()
            ax4.grid(True)

        plt.tight_layout()

        plot_path = os.path.join(self.save_dirs['model_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {plot_path}")
        plt.close()

    def train(self):
        # Setup data loaders
        train_transforms, val_transforms = self._get_transforms()

        db_train = db.FIREDatasetRandom(inputRes=(256, 256), transform=train_transforms, mode="train", num_frame=1)
        trainloader = DataLoader(db_train, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

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

                if batch_idx % 50 == 0:
                    print(f'Batch {batch_idx}/{len(trainloader)}, Loss: {loss.item():.4f}')

            avg_train_loss = epoch_train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            self.epochs.append(epoch + 1)

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