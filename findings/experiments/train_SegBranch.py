from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import imageio
import matplotlib.pyplot as plt
import json

from network.joint_pred_seg import SegBranch, SegDecoder, SegEncoder
from torchvision.models import resnet101, ResNet101_Weights

from dataloaders import custom_transforms as tr
from dataloaders import DAVIS_dataloader as davis
from dataloaders import FIRE_dataloader as db
from mypath import Path


class ModelConfig:
    """Configuration class for segmentation branch training"""

    def __init__(self, args, round_num=0, phase='fire'):
        self.gpu_id = 0
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.last_epoch = 0
        self.phase = phase

        # Adjust epochs and validation based on phase
        if phase == 'davis':
            self.num_epochs = args.davis_epochs
            self.do_validation = False  # No validation for Davis
        else:
            self.num_epochs = args.fire_epochs
            self.do_validation = True

        self.batch_size = 8
        self.snapshot = 10
        self.lr = 1e-3
        self.wd = 5e-4
        self.lr_decay = 0.9
        self.round_num = round_num

        # Model naming
        if phase == 'davis':
            self.model_name = f'SegBranch_DAVIS_pretrain_round{round_num}'
        elif phase == 'combined':
            self.model_name = f'SegBranch_DAVIS_FIRE_round{round_num}'
        else:
            self.model_name = f'SegBranch_FIRE_round{round_num}'

        # Paths
        self.save_dir = Path.save_root_dir()
        self.save_model_dir = os.path.join(self.save_dir, self.model_name)
        self.results_dir = os.path.join(self.save_dir, 'seg_branch_multiround', phase)

        # For loading Davis pretrained weights
        self.davis_pretrained_path = None
        if phase == 'fire' and args.use_davis_pretrain:
            self.davis_pretrained_path = os.path.join(
                self.save_dir,
                f'SegBranch_DAVIS_pretrain_round{round_num}',
                f'SegBranch_DAVIS_pretrain_round{round_num}_final.pth'
            )


def setup_directories(config):
    """Create necessary directories"""
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.save_model_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Create results visualization directory
    results_vis_dir = os.path.join(config.save_model_dir, 'results')
    os.makedirs(results_vis_dir, exist_ok=True)


def create_data_loaders(config, dataset_type='fire'):
    """Create training and validation data loaders"""

    if dataset_type == 'davis':
        print("Loading DAVIS dataset for pretraining...")
        # Note: Davis dataloader needs to be used differently since it doesn't have single frames
        # We'll use the current frame from the sequence
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
        ])

        train_set = davis.DAVISDataset(
            inputRes=(256, 256),
            samples_list_file='/home/c43n256/STCNN/data/DAVIS16_samples_list.txt',
            transform=composed_transforms,
            num_frame=1  # Use single frame for segmentation
        )

        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        val_loader = None

    else:  # fire
        print("Loading FIRE dataset...")
        train_set = db.FIREDataset(
            inputRes=(256, 256),
            samples_path="/home/c43n256/Data/Mask_Data",
            transform=None,
            mode="train",
            num_frame=1
        )

        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )

        if config.do_validation:
            val_set = db.FIREDataset(
                inputRes=(256, 256),
                samples_path="/home/c43n256/Data/Mask_Data",
                transform=None,
                mode="test",
                num_frame=1
            )
            val_loader = DataLoader(
                val_set,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4
            )
        else:
            val_loader = None

    return train_loader, val_loader


def initialize_model(config):
    """Initialize segmentation model"""
    encoder = SegEncoder()

    # Load pretrained ResNet101 weights
    print("Initializing encoder with ResNet101 ImageNet weights...")
    pretrained_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained_model.state_dict()
    model_dict = encoder.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    encoder.load_state_dict(model_dict)

    decoder = SegDecoder()
    net = SegBranch(net_enc=encoder, net_dec=decoder)

    # Load Davis pretrained weights if available
    if config.davis_pretrained_path and os.path.exists(config.davis_pretrained_path):
        print(f"Loading Davis pretrained weights from: {config.davis_pretrained_path}")
        try:
            checkpoint = torch.load(config.davis_pretrained_path, map_location=config.device)
            net.load_state_dict(checkpoint)
            print("Successfully loaded Davis pretrained weights")
        except Exception as e:
            print(f"Error loading Davis weights: {e}")
            print("Continuing with ImageNet initialization")

    net.to(config.device)

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return net


def validate(net, val_loader, criterion, device):
    """Run validation and return average loss"""
    net.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for sample_batched in val_loader:
            inputs, gts = sample_batched['frame'], sample_batched['seg_gt']
            inputs, gts = inputs.to(device), gts.to(device)

            pred = net.forward(inputs)

            # Multi-scale loss
            loss = criterion(pred[-1], gts)
            for i in reversed(range(len(pred) - 1)):
                loss = loss + 1 * criterion(pred[i], gts)

            val_loss += loss.item()
            num_batches += 1

    net.train()
    return val_loss / num_batches


def save_visualization(net, val_loader, save_path, device):
    """Save a visualization sample"""
    net.eval()
    with torch.no_grad():
        sample_batched = next(iter(val_loader))
        inputs, gts = sample_batched['frame'], sample_batched['seg_gt']
        inputs, gts = inputs.to(device), gts.to(device)
        pred = net.forward(inputs)

        inputs_vis = inputs[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
        inputs_vis = (inputs_vis - inputs_vis.min()) / max((inputs_vis.max() - inputs_vis.min()), 1e-8) * 255

        gt = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0]) * 255
        gt = np.concatenate([gt, gt, gt], axis=2)

        samples = pred[-1][0, :, :, :].data.cpu().numpy()
        samples = 1 / (1 + np.exp(-samples))
        samples = samples.transpose([1, 2, 0]) * 255
        samples = np.concatenate([samples, samples, samples], axis=2)

        samples = np.concatenate((samples, gt, inputs_vis), axis=0)
        imageio.imwrite(save_path, samples.astype(np.uint8))
    net.train()


def run_single_round(args, round_num, phase='fire'):
    """Run a single training round"""
    print(f"\n{'=' * 80}")
    print(f"STARTING ROUND {round_num + 1}/3 - PHASE: {phase.upper()}")
    print(f"{'=' * 80}\n")

    config = ModelConfig(args, round_num=round_num, phase=phase)

    if torch.cuda.is_available():
        print(f"Using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    else:
        print("Using CPU")

    setup_directories(config)

    # Create data loaders
    dataset_type = 'davis' if phase == 'davis' else 'fire'
    train_loader, val_loader = create_data_loaders(config, dataset_type)

    # Initialize model
    net = initialize_model(config)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().to(config.device)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * config.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': config.lr, 'weight_decay': config.wd}
    ], momentum=0.9)

    # TensorBoard
    log_dir = os.path.join(
        config.save_dir,
        'SegBranch_runs',
        f'{phase}_round{round_num}_' + datetime.now().strftime('%b%d_%H-%M-%S')
    )
    writer = SummaryWriter(log_dir=log_dir, comment=f'-{phase}-round{round_num}')

    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = -1

    print(f"Starting {phase.upper()} training for round {round_num + 1}...")
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")

    for epoch in range(config.num_epochs):
        start_time = timeit.default_timer()
        epoch_train_loss = 0.0
        num_train_batches = 0

        net.train()

        for ii, sample_batched in enumerate(train_loader):
            # Learning rate decay
            current_progress = (epoch * len(train_loader) + ii) / (config.num_epochs * len(train_loader))
            optimizer.param_groups[0]['lr'] = 2 * config.lr * (1 - current_progress) ** config.lr_decay
            optimizer.param_groups[1]['lr'] = config.lr * (1 - current_progress) ** config.lr_decay

            # Get data
            inputs, gts = sample_batched['frame'], sample_batched['seg_gt']
            inputs.requires_grad_()
            inputs, gts = inputs.to(config.device), gts.to(config.device)

            # Forward pass
            pred = net.forward(inputs)

            # Multi-scale loss
            optimizer.zero_grad()
            loss = criterion(pred[-1], gts)
            for i in reversed(range(len(pred) - 1)):
                loss = loss + 1 * criterion(pred[i], gts)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

            if ii % 10 == 0:
                print(f"Epoch: [{epoch + 1}/{config.num_epochs}], "
                      f"Batch: [{ii}/{len(train_loader)}], "
                      f"Loss: {loss.item():.8f}, "
                      f"LR: {optimizer.param_groups[1]['lr']:.8f}")

        # Average training loss
        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # Validation
        avg_val_loss = 0
        if config.do_validation and val_loader is not None:
            avg_val_loss = validate(net, val_loader, criterion, config.device)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_path = os.path.join(config.save_model_dir, f'{config.model_name}_best.pth')
                torch.save(net.state_dict(), best_path)

        # Logging
        print(f"{phase.upper()} Round {round_num + 1} - Epoch [{epoch + 1}/{config.num_epochs}] - "
              f"Train Loss: {avg_train_loss:.6f}" +
              (f", Val Loss: {avg_val_loss:.6f}" if config.do_validation else ""))

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        if config.do_validation:
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[1]['lr'], epoch)

        # Save visualization
        if val_loader and (epoch + 1) % 5 == 0:
            vis_path = os.path.join(config.save_model_dir, 'results', f'epoch_{epoch + 1}.png')
            save_visualization(net, val_loader, vis_path, config.device)

        # Save checkpoints
        if (epoch + 1) % config.snapshot == 0:
            checkpoint_path = os.path.join(config.save_model_dir,
                                           f'{config.model_name}_epoch{epoch + 1}.pth')
            torch.save(net.state_dict(), checkpoint_path)

    # Save final model
    final_path = os.path.join(config.save_model_dir, f'{config.model_name}_final.pth')
    torch.save(net.state_dict(), final_path)

    writer.close()

    # Save results
    results = {
        'round': round_num,
        'phase': phase,
        'train_losses': train_losses,
        'val_losses': val_losses if config.do_validation else [],
        'best_val_loss': best_val_loss if config.do_validation else None,
        'best_epoch': best_epoch if config.do_validation else None,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1] if val_losses else None
    }

    results_path = os.path.join(config.results_dir, f'round{round_num}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Plot training curves
    plot_training_curves(train_losses, val_losses if config.do_validation else None,
                         os.path.join(config.save_model_dir, 'training_curve.png'),
                         phase, round_num)

    return results


def plot_training_curves(train_losses, val_losses, save_path, phase, round_num):
    """Plot and save training curves"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Seg Branch - {phase.upper()} Round {round_num + 1}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def analyze_multi_round_results(results_list, save_dir, phase):
    """Analyze results across multiple rounds"""
    print(f"\n{'=' * 80}")
    print(f"STATISTICAL ANALYSIS - SEG BRANCH {phase.upper()}")
    print(f"{'=' * 80}\n")

    final_train_losses = [r['final_train_loss'] for r in results_list]

    stats_summary = {
        'phase': phase,
        'model': 'SegBranch',
        'final_train_loss': {
            'mean': np.mean(final_train_losses),
            'std': np.std(final_train_losses),
            'min': np.min(final_train_losses),
            'max': np.max(final_train_losses),
            'values': final_train_losses
        }
    }

    if results_list[0]['final_val_loss'] is not None:
        final_val_losses = [r['final_val_loss'] for r in results_list]
        best_val_losses = [r['best_val_loss'] for r in results_list]

        stats_summary['final_val_loss'] = {
            'mean': np.mean(final_val_losses),
            'std': np.std(final_val_losses),
            'min': np.min(final_val_losses),
            'max': np.max(final_val_losses),
            'values': final_val_losses
        }

        stats_summary['best_val_loss'] = {
            'mean': np.mean(best_val_losses),
            'std': np.std(best_val_losses),
            'min': np.min(best_val_losses),
            'max': np.max(best_val_losses),
            'values': best_val_losses
        }

    print("Final Training Loss:")
    print(f"  Mean ± Std: {stats_summary['final_train_loss']['mean']:.6f} ± "
          f"{stats_summary['final_train_loss']['std']:.6f}")
    print(f"  Values: {final_train_losses}\n")

    if 'final_val_loss' in stats_summary:
        print("Final Validation Loss:")
        print(f"  Mean ± Std: {stats_summary['final_val_loss']['mean']:.6f} ± "
              f"{stats_summary['final_val_loss']['std']:.6f}")
        print(f"  Values: {final_val_losses}\n")

        print("Best Validation Loss:")
        print(f"  Mean ± Std: {stats_summary['best_val_loss']['mean']:.6f} ± "
              f"{stats_summary['best_val_loss']['std']:.6f}")
        print(f"  Values: {best_val_losses}\n")

    stats_path = os.path.join(save_dir, f'{phase}_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=4)

    plot_multi_round_comparison(results_list, save_dir, phase)

    return stats_summary


def plot_multi_round_comparison(results_list, save_dir, phase):
    """Create comparison plots for multiple rounds"""
    has_val = results_list[0]['final_val_loss'] is not None

    plt.figure(figsize=(15, 5) if has_val else (10, 5))

    if has_val:
        plt.subplot(1, 2, 1)

    for i, results in enumerate(results_list):
        epochs = range(1, len(results['train_losses']) + 1)
        plt.plot(epochs, results['train_losses'], label=f'Round {i + 1}', alpha=0.7)

    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Seg Branch Training Loss - {phase.upper()}')
    plt.legend()
    plt.grid(True)

    if has_val:
        plt.subplot(1, 2, 2)
        for i, results in enumerate(results_list):
            epochs = range(1, len(results['val_losses']) + 1)
            plt.plot(epochs, results['val_losses'], label=f'Round {i + 1}', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title(f'Seg Branch Validation Loss - {phase.upper()}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{phase}_comparison.png'), dpi=300)
    plt.close()


def main(args):
    """Main function"""
    print(f"\n{'=' * 80}")
    print("SEGMENTATION BRANCH MULTI-ROUND TRAINING")
    print(f"Training mode: {args.training_mode}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"{'=' * 80}\n")

    save_dir = os.path.join(Path.save_root_dir(), 'seg_branch_multiround')

    if args.training_mode == 'davis_only':
        print("Mode: Davis Pretraining Only\n")
        results_list = []
        for round_num in range(args.num_rounds):
            try:
                results = run_single_round(args, round_num, phase='davis')
                results_list.append(results)
                print(f"\nDavis Round {round_num + 1} completed!")
                print(f"Final Train Loss: {results['final_train_loss']:.6f}")
            except Exception as e:
                print(f"\nError in Davis round {round_num + 1}: {str(e)}")
                import traceback
                traceback.print_exc()

        if len(results_list) == args.num_rounds:
            analyze_multi_round_results(results_list, os.path.join(save_dir, 'davis'), 'davis')

    elif args.training_mode == 'fire_only':
        print("Mode: FIRE Training Only\n")
        results_list = []
        for round_num in range(args.num_rounds):
            try:
                results = run_single_round(args, round_num, phase='fire')
                results_list.append(results)
                print(f"\nFIRE Round {round_num + 1} completed!")
                print(f"Final Train Loss: {results['final_train_loss']:.6f}")
                print(f"Final Val Loss: {results['final_val_loss']:.6f}")
                print(f"Best Val Loss: {results['best_val_loss']:.6f}")
            except Exception as e:
                print(f"\nError in FIRE round {round_num + 1}: {str(e)}")
                import traceback
                traceback.print_exc()

        if len(results_list) == args.num_rounds:
            analyze_multi_round_results(results_list, os.path.join(save_dir, 'fire'), 'fire')

    elif args.training_mode == 'combined':
        print("Mode: Davis Pretraining + FIRE Training\n")
        davis_results = []
        fire_results = []

        for round_num in range(args.num_rounds):
            print(f"\n{'=' * 80}")
            print(f"ROUND {round_num + 1}/{args.num_rounds}")
            print(f"{'=' * 80}\n")

            try:
                print(f"Phase 1: Davis Pretraining")
                davis_result = run_single_round(args, round_num, phase='davis')
                davis_results.append(davis_result)
                print(f"\nDavis completed - Loss: {davis_result['final_train_loss']:.6f}")
            except Exception as e:
                print(f"Error in Davis: {e}")
                import traceback
                traceback.print_exc()
                continue

            try:
                print(f"\nPhase 2: FIRE Training")
                fire_result = run_single_round(args, round_num, phase='fire')
                fire_results.append(fire_result)
                print(f"\nFIRE completed - Train: {fire_result['final_train_loss']:.6f}, "
                      f"Val: {fire_result['final_val_loss']:.6f}")
            except Exception as e:
                print(f"Error in FIRE: {e}")
                import traceback
                traceback.print_exc()

        if len(davis_results) == args.num_rounds:
            analyze_multi_round_results(davis_results,
                                        os.path.join(save_dir, 'combined_davis'), 'davis')

        if len(fire_results) == args.num_rounds:
            analyze_multi_round_results(fire_results,
                                        os.path.join(save_dir, 'combined_fire'), 'fire')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-round Segmentation Branch training with Davis pretraining")

    parser.add_argument("--num_rounds", type=int, default=3,
                        help="Number of training rounds")

    parser.add_argument("--training_mode", type=str, default="fire_only",
                        choices=["davis_only", "fire_only", "combined"],
                        help="Training mode")

    parser.add_argument("--davis_epochs", type=int, default=50,
                        help="Epochs for Davis pretraining")

    parser.add_argument("--fire_epochs", type=int, default=200,
                        help="Epochs for FIRE training")

    parser.add_argument("--use_davis_pretrain", action="store_true",
                        help="Use Davis pretrained weights")

    args = parser.parse_args()

    if args.training_mode == 'combined':
        args.use_davis_pretrain = True

    main(args)