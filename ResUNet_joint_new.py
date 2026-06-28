from __future__ import absolute_import, division, print_function

import argparse
import os
from datetime import datetime
import socket
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from network.ResUNet_new import ResNetEncoder, ResNetDecoder, ResUNet
from dataloaders import custom_transforms as tr
from dataloaders import FIRE_dataloader as db
from mypath import Path


class ModelConfig:
    """Configuration class to hold all model parameters"""

    def __init__(self, args):
        self.gpu_id = 0
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.resume_epoch = 0
        self.nEpochs = 201
        self.batch_size = 8
        self.snapshot = 10
        self.seg_lr = 1e-4
        self.wd = 5e-4
        self.lr_decay = 0.9
        self.lr_decay_freq = 50
        self.num_frame = 4  # Single frame for segmentation
        self.model_name = 'ResUNet_Pretrained'
        self.do_validation = args.do_validation
        self.use_attention = args.use_attention

        # ResUNet architecture parameters
        self.encoder_name = "resnet34"
        self.encoder_weights = "imagenet"
        self.in_channels = 3
        self.classes = 1
        self.decoder_channels = (256, 128, 64)

        # Paths
        self.save_dir = Path.save_root_dir()
        self.save_model_dir = os.path.join(self.save_dir, self.model_name)

        # Pretrained weights (optional)
        self.pretrained_resunet_path = None  # Set this if you have pretrained ResUNet
        self.resume_model_path = None  # Set this to resume training


class ModelInitializer:
    """Handles model initialization and weight loading"""

    @staticmethod
    def load_resunet_weights(net, checkpoint_path, device, use_attention=True):
        """
        Load ResUNet weights with smart handling of attention blocks.

        Args:
            net: ResUNet model instance
            checkpoint_path: Path to checkpoint file
            device: torch device
            use_attention: Whether current model has attention blocks

        Returns:
            bool: Success status
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at: {checkpoint_path}")
            return False

        try:
            print(f"Loading ResUNet weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Extract state_dict if nested
            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint

            model_dict = net.state_dict()

            print(f"\nCheckpoint has {len(pretrained_dict)} keys")
            print(f"Current model has {len(model_dict)} keys")
            print("Sample checkpoint keys:", list(pretrained_dict.keys())[:5])
            print("Sample model keys:", list(model_dict.keys())[:5])

            # Smart weight matching
            compatible_weights = {}
            attention_keys_skipped = []
            shape_mismatches = []

            for k, v in model_dict.items():
                # Check if key exists in checkpoint
                if k in pretrained_dict:
                    pretrained_v = pretrained_dict[k]

                    # Check shape compatibility
                    if v.shape == pretrained_v.shape:
                        compatible_weights[k] = pretrained_v
                    else:
                        shape_mismatches.append((k, v.shape, pretrained_v.shape))

                # Handle attention blocks (new layers not in old checkpoint)
                elif 'attention' in k.lower():
                    attention_keys_skipped.append(k)

            # Try pattern matching for different checkpoint formats
            if len(compatible_weights) < len(model_dict) * 0.3:  # Less than 30% match
                print("\nTrying alternative key matching patterns...")

                # Pattern 1: Remove 'encoder.' or 'decoder.' prefix
                for prefix in ['encoder.', 'decoder.', 'module.', 'seg.']:
                    alt_dict = {}
                    for k in model_dict.keys():
                        # Try with prefix
                        alt_key = prefix + k
                        if alt_key in pretrained_dict and model_dict[k].shape == pretrained_dict[alt_key].shape:
                            alt_dict[k] = pretrained_dict[alt_key]

                        # Try removing prefix
                        if k.startswith(prefix):
                            alt_key = k[len(prefix):]
                            if alt_key in pretrained_dict and model_dict[k].shape == pretrained_dict[alt_key].shape:
                                alt_dict[k] = pretrained_dict[alt_key]

                    if len(alt_dict) > len(compatible_weights):
                        compatible_weights = alt_dict
                        print(f"Pattern '{prefix}' matched {len(alt_dict)} keys")

            # Load compatible weights
            if compatible_weights:
                model_dict.update(compatible_weights)
                missing_keys, unexpected_keys = net.load_state_dict(model_dict, strict=False)

                print(f"\n✓ Successfully loaded {len(compatible_weights)}/{len(model_dict)} weights")

                if attention_keys_skipped:
                    print(f"\n⚠ Skipped {len(attention_keys_skipped)} attention block keys (new layers):")
                    for k in attention_keys_skipped[:5]:
                        print(f"  - {k}")
                    if len(attention_keys_skipped) > 5:
                        print(f"  ... and {len(attention_keys_skipped) - 5} more")

                if shape_mismatches:
                    print(f"\n⚠ Found {len(shape_mismatches)} shape mismatches:")
                    for k, model_shape, ckpt_shape in shape_mismatches[:3]:
                        print(f"  - {k}: model {model_shape} vs checkpoint {ckpt_shape}")
                    if len(shape_mismatches) > 3:
                        print(f"  ... and {len(shape_mismatches) - 3} more")

                if missing_keys:
                    print(f"\n⚠ Missing keys ({len(missing_keys)}):")
                    for k in list(missing_keys)[:5]:
                        print(f"  - {k}")
                    if len(missing_keys) > 5:
                        print(f"  ... and {len(missing_keys) - 5} more")

                return True
            else:
                print("\n✗ No compatible weights found!")
                return False

        except Exception as e:
            print(f"Error loading ResUNet weights: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def load_encoder_only(encoder, checkpoint_path, device):
        """Load only encoder weights (for transfer learning scenarios)"""
        if not os.path.exists(checkpoint_path):
            print(f"Encoder checkpoint not found at: {checkpoint_path}")
            return False

        try:
            print(f"Loading encoder weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint

            # Filter encoder weights
            encoder_dict = encoder.state_dict()
            encoder_pretrained = {}

            for k, v in pretrained_dict.items():
                # Try different patterns
                encoder_key = k
                if k.startswith('encoder.'):
                    encoder_key = k[8:]  # Remove 'encoder.' prefix
                elif k.startswith('module.encoder.'):
                    encoder_key = k[15:]

                if encoder_key in encoder_dict and v.shape == encoder_dict[encoder_key].shape:
                    encoder_pretrained[encoder_key] = v

            if encoder_pretrained:
                encoder_dict.update(encoder_pretrained)
                encoder.load_state_dict(encoder_dict, strict=False)
                print(f"✓ Loaded {len(encoder_pretrained)}/{len(encoder_dict)} encoder weights")
                return True
            else:
                print("✗ No compatible encoder weights found")
                return False

        except Exception as e:
            print(f"Error loading encoder weights: {e}")
            return False


def setup_directories(config):
    """Create necessary directories"""
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.save_model_dir, exist_ok=True)


def create_data_loaders(config):
    """Create training and validation data loaders"""
    # Define augmentation transformations
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
    ])

    # Training Dataset
    train_set = db.FIREDatasetRandom(
        inputRes=(256, 256),
        mode="train",
        num_frame=config.num_frame,
        transform=composed_transforms
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Test dataset (only create if validation is enabled)
    test_loader = None
    if config.do_validation:
        test_set = db.FIREDatasetRandom(
            inputRes=(256, 256),
            mode="test",
            num_frame=config.num_frame
        )
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            num_workers=4,
            shuffle=False
        )

    return train_loader, test_loader


def create_models_and_optimizers(config):
    """Create models, loss functions, and optimizers"""
    # Create ResUNet model
    net = ResUNet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights if config.resume_epoch == 0 else None,
        in_channels=config.in_channels,
        n_classes=config.classes,
        decoder_channels=config.decoder_channels,
        use_attention=config.use_attention
    )

    # Initialize models
    initializer = ModelInitializer()

    if config.resume_epoch == 0:
        # Load pretrained weights if available
        if config.pretrained_resunet_path:
            initializer.load_resunet_weights(
                net,
                config.pretrained_resunet_path,
                config.device,
                use_attention=config.use_attention
            )
        else:
            print("Training from ImageNet pretrained encoder + random decoder")
    else:
        # Resume from checkpoint
        if config.resume_model_path and os.path.exists(config.resume_model_path):
            print(f"Resuming from: {config.resume_model_path}")
            checkpoint = torch.load(config.resume_model_path, map_location=config.device)
            if 'state_dict' in checkpoint:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net.load_state_dict(checkpoint)
        else:
            print(f"Resume path not found: {config.resume_model_path}")

    # Move model to device
    net.to(config.device)

    # Print model info
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in net.encoder.parameters())
    decoder_params = sum(p.numel() for p in net.decoder.parameters())

    print("\nModel Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss function
    seg_criterion = nn.BCEWithLogitsLoss().to(config.device)

    # Optimizer with different learning rates for encoder and decoder
    optimizer = optim.Adam([
        {'params': net.encoder.parameters(), 'lr': config.seg_lr * 0.1},  # Lower LR for pretrained encoder
        {'params': net.decoder.parameters(), 'lr': config.seg_lr}
    ], weight_decay=config.wd)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_decay_freq,
        gamma=config.lr_decay
    )

    return net, seg_criterion, optimizer, scheduler


def calculate_metrics(predictions, targets, threshold=0.5):
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


def train_epoch(net, data_loader, optimizer, criterion, config, epoch, writer):
    """Train for one epoch"""
    net.train()

    epoch_loss = 0
    epoch_iou = 0
    epoch_pa = 0
    num_batches = len(data_loader)
    start_time = timeit.default_timer()

    for ii, sample_batched in enumerate(data_loader):
        inputs = sample_batched['images'].to(config.device)
        gts = sample_batched['seg_gt'].to(config.device).float()

        # Forward pass
        optimizer.zero_grad()
        outputs = net(inputs)

        # Calculate loss
        loss = criterion(outputs, gts)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        iou, pa = calculate_metrics(outputs, gts)

        epoch_loss += loss.item()
        epoch_iou += iou
        epoch_pa += pa

        # Logging
        if ii % 50 == 0:
            print(f"Epoch [{epoch + 1}] Batch [{ii}/{num_batches}] "
                  f"Loss: {loss.item():.4f} IoU: {iou:.4f} PA: {pa:.4f}")

    avg_loss = epoch_loss / num_batches
    avg_iou = epoch_iou / num_batches
    avg_pa = epoch_pa / num_batches

    epoch_time = timeit.default_timer() - start_time

    return avg_loss, avg_iou, avg_pa, epoch_time


def validate(net, data_loader, criterion, config):
    """Validate the model"""
    net.eval()

    val_loss = 0
    val_iou = 0
    val_pa = 0

    with torch.no_grad():
        for sample in data_loader:
            inputs = sample['images'].to(config.device)
            gts = sample['seg_gt'].to(config.device).float()

            outputs = net(inputs)
            loss = criterion(outputs, gts)
            iou, pa = calculate_metrics(outputs, gts)

            val_loss += loss.item()
            val_iou += iou
            val_pa += pa

    num_batches = len(data_loader)
    return val_loss / num_batches, val_iou / num_batches, val_pa / num_batches


def save_checkpoint(net, optimizer, epoch, config, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': {
            'encoder_name': config.encoder_name,
            'in_channels': config.in_channels,
            'classes': config.classes,
            'decoder_channels': config.decoder_channels,
            'use_attention': config.use_attention,
        }
    }

    checkpoint_path = os.path.join(config.save_model_dir, f'{config.model_name}_epoch-{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    if is_best:
        best_path = os.path.join(config.save_model_dir, f'{config.model_name}_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")


def main(args):
    """Main training function"""
    # Initialize configuration
    config = ModelConfig(args)

    if torch.cuda.is_available():
        print(f"CUDA available, using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    else:
        print("CUDA not available, using CPU.")

    # Setup directories
    setup_directories(config)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(config)
    print(f"\nDataset Info:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    if test_loader:
        print(f"  Validation samples: {len(test_loader.dataset)}")

    # Create models and optimizers
    net, criterion, optimizer, scheduler = create_models_and_optimizers(config)

    # Setup logging
    log_dir = os.path.join(
        config.save_dir,
        'ResUNet_runs',
        datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()
    )
    writer = SummaryWriter(log_dir=log_dir, comment='-resunet')

    # Training loop
    train_losses = []
    train_ious = []
    train_pas = []
    val_losses = []
    val_ious = []
    val_pas = []

    best_val_iou = 0.0

    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    for epoch in range(config.resume_epoch, config.nEpochs):
        # Training
        train_loss, train_iou, train_pa, epoch_time = train_epoch(
            net, train_loader, optimizer, criterion, config, epoch, writer
        )

        train_losses.append(train_loss)
        train_ious.append(train_iou)
        train_pas.append(train_pa)

        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch [{epoch + 1}/{config.nEpochs}] Summary:")
        print(f"  Train Loss: {train_loss:.6f} | IoU: {train_iou:.4f} | PA: {train_pa:.4f}")
        print(f"  Learning Rate: {current_lr:.6f} | Time: {epoch_time:.2f}s")

        # Validation
        if config.do_validation and test_loader is not None:
            val_loss, val_iou, val_pa = validate(net, test_loader, criterion, config)
            val_losses.append(val_loss)
            val_ious.append(val_iou)
            val_pas.append(val_pa)

            print(f"  Val Loss: {val_loss:.6f} | IoU: {val_iou:.4f} | PA: {val_pa:.4f}")

            # Save best model
            is_best = val_iou > best_val_iou
            if is_best:
                best_val_iou = val_iou
                print(f"  ★ New best IoU: {best_val_iou:.4f}")

        # Tensorboard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('IoU/Train', train_iou, epoch)
        writer.add_scalar('PixelAccuracy/Train', train_pa, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        if config.do_validation and val_losses:
            writer.add_scalar('Loss/Val', val_losses[-1], epoch)
            writer.add_scalar('IoU/Val', val_ious[-1], epoch)
            writer.add_scalar('PixelAccuracy/Val', val_pas[-1], epoch)

        # Save checkpoint
        if (epoch % config.snapshot) == config.snapshot - 1 and epoch != 0:
            save_checkpoint(net, optimizer, epoch, config,
                            is_best=(val_iou == best_val_iou if config.do_validation else False))

        print("-" * 60)

    # Final checkpoint
    save_checkpoint(net, optimizer, config.nEpochs - 1, config)

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_ious, label='Train IoU')
    if val_ious:
        plt.plot(val_ious, label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(train_pas, label='Train PA')
    if val_pas:
        plt.plot(val_pas, label='Val PA')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.title('Pixel Accuracy Curves')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.save_model_dir, f'training_curves_{config.model_name}.png'), dpi=300)

    writer.close()
    print("\n" + "=" * 60)
    print("Training completed!")
    if config.do_validation:
        print(f"Best validation IoU: {best_val_iou:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for ResUNet segmentation")
    parser.add_argument("--do_validation", action="store_true", help="Do validation during training")
    parser.add_argument("--use_attention", action="store_true", default=True, help="Use attention gates in decoder")

    args = parser.parse_args()
    main(args)