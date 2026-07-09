from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ResT.models.rest import *
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import json

from sklearn.metrics import roc_auc_score, f1_score

from network.joint_pred_seg import SegBranch, SegDecoder, SegEncoder, STCNN, FramePredEncoder, FramePredDecoder, \
    JointSegDecoder

from network.googlenet import Inception3

from network.shuffle import PretrainedShuffleEncoder

from dataloaders import custom_transforms as tr
from dataloaders import DAVIS_dataloader as davis
from dataloaders import FIRE_dataloader as db
from mypath import Path


class ModelConfig:
    """Configuration class to hold all model parameters"""

    def __init__(self, args, round_num=0):
        self.gpu_id = 0
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.resume_epoch = 0
        self.nEpochs = 100  # Changed to 100 epochs per round
        self.batch_size = 6
        self.snapshot = 10  # Save every 10 epochs
        self.pred_lr = 1e-8
        self.seg_lr = 1e-4
        self.lr_D = 1e-4
        self.wd = 5e-4
        self.beta = 0.001
        self.margin = 0.3
        self.num_frame = args.frame_nums
        self.model_name = f'STCNN_frame_NO_DAVIS{self.num_frame}_round{round_num}'
        self.do_validation = True  # Always do validation
        self.round_num = round_num

        # Paths
        self.save_dir = Path.save_root_dir()
        self.save_model_dir = os.path.join(self.save_dir, self.model_name)
        self.results_dir = os.path.join(self.save_dir, 'multi_round_results')
        self.pretrained_netd_path = '/home/c43n256/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4/NetD_epoch-99.pth'
        self.pretrained_netg_path = '/home/c43n256/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4/NetG_epoch-99.pth'
        self.pretrained_seg_path = '/home/c43n256/STCNN/output/Seg_Branch_NEW_RUN/Seg_Branch_NEW_RUN_epoch-11999.pth'


class ModelInitializer:
    """Handles model initialization and weight loading"""

    @staticmethod
    def initialize_netd(netd, model_path, device):
        """Initialize discriminator with pretrained weights"""
        try:
            if os.path.exists(model_path):
                print(f"Loading NetD weights from: {model_path}")
                state_dict = torch.load(model_path, map_location=device)
                netd.load_state_dict(state_dict)
            else:
                print("Saved NetD weights not found, using torchvision pretrained weights")
                hub_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
                hub_model.eval()

                pretrained_dict = hub_model.state_dict()
                model_dict = netd.state_dict()

                filtered_dict = {k: v for k, v in pretrained_dict.items()
                                 if k in model_dict and not k.startswith('fc.')}

                model_dict.update(filtered_dict)
                netd.load_state_dict(model_dict)

        except Exception as e:
            print(f"Error loading NetD weights: {e}")
            print("Using random initialization for NetD")

    @staticmethod
    def load_prediction_weights(pred_enc, pred_dec, netg_path, device):
        """Load prediction encoder and decoder weights"""
        if not os.path.exists(netg_path):
            print(f"NetG weights not found at: {netg_path}")
            return False

        try:
            print(f"Loading prediction weights from: {netg_path}")
            pretrained_netg_dict = torch.load(netg_path, map_location=device)

            pred_enc_dict = pred_enc.state_dict()
            pred_enc_pretrained = {k: v for k, v in pretrained_netg_dict.items() if k in pred_enc_dict}
            pred_enc_dict.update(pred_enc_pretrained)
            pred_enc.load_state_dict(pred_enc_dict)
            print(f"Loaded {len(pred_enc_pretrained)}/{len(pred_enc_dict)} encoder weights")

            pred_dec_dict = pred_dec.state_dict()
            pred_dec_pretrained = {k: v for k, v in pretrained_netg_dict.items() if k in pred_dec_dict}
            pred_dec_dict.update(pred_dec_pretrained)
            pred_dec.load_state_dict(pred_dec_dict)
            print(f"Loaded {len(pred_dec_pretrained)}/{len(pred_dec_dict)} decoder weights")

            return True
        except Exception as e:
            print(f"Error loading prediction weights: {e}")
            return False

    @staticmethod
    def load_segmentation_weights(seg_enc, seg_path, device):
        """Load segmentation encoder weights"""
        if not os.path.exists(seg_path):
            print(f"Segmentation weights not found at: {seg_path}")
            return False

        try:
            print(f"Loading segmentation weights from: {seg_path}")
            checkpoint = torch.load(seg_path, map_location=device)

            if 'state_dict' in checkpoint:
                pretrained_seg_dict = checkpoint['state_dict']
            else:
                pretrained_seg_dict = checkpoint

            seg_enc_dict = seg_enc.state_dict()

            print(f"\nPretrained model has {len(pretrained_seg_dict)} keys")
            print("Sample pretrained keys:", list(pretrained_seg_dict.keys())[:5])

            encoder_dict = {}

            pattern1 = {k[8:]: v for k, v in pretrained_seg_dict.items() if k.startswith("encoder.")}
            pattern2 = {k[4:]: v for k, v in pretrained_seg_dict.items() if k.startswith("seg.")}
            pattern3 = {k[7:]: v for k, v in pretrained_seg_dict.items() if k.startswith("module.")}
            pattern4 = pretrained_seg_dict

            for pattern_name, pattern in [("encoder.", pattern1), ("seg.", pattern2),
                                          ("module.", pattern3), ("no prefix", pattern4)]:
                matches = sum(1 for k in seg_enc_dict.keys() if k in pattern and 'TempAttention' not in k)
                print(f"{pattern_name}: {matches} potential matches")
                if matches > len(encoder_dict):
                    encoder_dict = pattern

            compatible_weights = {}

            for k, v in seg_enc_dict.items():
                if 'TempAttention' in k:
                    continue

                if k in encoder_dict:
                    if v.shape == encoder_dict[k].shape:
                        compatible_weights[k] = encoder_dict[k]

            if compatible_weights:
                seg_enc_dict.update(compatible_weights)
                seg_enc.load_state_dict(seg_enc_dict, strict=False)
                print(f"\nSuccessfully loaded {len(compatible_weights)}/{len(seg_enc_dict)} weights")
            else:
                print("\nNo compatible weights found!")

            return len(compatible_weights) > 0

        except Exception as e:
            print(f"Error loading segmentation weights: {e}")
            import traceback
            traceback.print_exc()
            return False


def setup_directories(config):
    """Create necessary directories"""
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.save_model_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)


def create_data_loaders(config):
    """Create training and validation data loaders"""
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
    ])

    train_set = db.FIREDataset(
        inputRes=(256, 256),
        mode="train",
        num_frame=config.num_frame
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    test_set = db.FIREDataset(
        inputRes=(256, 256),
        mode="test",
        num_frame=config.num_frame
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=4,
        shuffle=False  # Don't shuffle test set for consistency
    )

    return train_loader, test_loader


def create_models_and_optimizers(config):
    """Create models, loss functions, and optimizers"""
    netd = Inception3(num_classes=1, aux_logits=False, transform_input=True)
    pred_enc = FramePredEncoder(frame_nums=config.num_frame)
    pred_dec = FramePredDecoder()
    seg_enc = SegEncoder()
    seg_dec = JointSegDecoder()

    initializer = ModelInitializer()

    initializer.initialize_netd(netd, config.pretrained_netd_path, config.device)
    initializer.load_prediction_weights(pred_enc, pred_dec, config.pretrained_netg_path, config.device)
    initializer.load_segmentation_weights(seg_enc, config.pretrained_seg_path, config.device)

    net = STCNN(pred_enc, pred_dec, seg_enc, seg_dec)

    net.to(config.device)
    netd.to(config.device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    lp_function = nn.MSELoss().to(config.device)
    criterion = nn.BCELoss().to(config.device)
    seg_criterion = nn.BCEWithLogitsLoss().to(config.device)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.seg_decoder.named_parameters()], 'lr': config.seg_lr},
        {'params': [param for name, param in net.seg_encoder.named_parameters()], 'lr': config.seg_lr},
    ], weight_decay=config.wd, momentum=0.9)

    optimizer_g = optim.Adam([
        {'params': [param for name, param in net.pred_encoder.named_parameters()], 'lr': config.pred_lr},
        {'params': [param for name, param in net.pred_decoder.named_parameters()], 'lr': config.pred_lr},
    ], lr=config.pred_lr, weight_decay=config.wd)

    optimizer_d = optim.Adam(netd.parameters(), lr=config.lr_D, weight_decay=config.wd)

    return net, netd, (lp_function, criterion, seg_criterion), (optimizer, optimizer_g, optimizer_d)


def train_epoch(net, netd, data_loader, optimizers, criteria, config, epoch, writer):
    """Train for one epoch"""
    optimizer, optimizer_g, optimizer_d = optimizers
    lp_function, criterion, seg_criterion = criteria

    net.train()
    netd.train()

    epoch_loss = 0
    num_batches = len(data_loader)
    start_time = timeit.default_timer()

    update_d = True
    update_g = False

    for ii, sample_batched in enumerate(data_loader):
        seqs = sample_batched['images'].to(config.device).requires_grad_()
        frames = sample_batched['frame'].to(config.device).requires_grad_()
        gts = sample_batched['seg_gt'].to(config.device)
        pred_gts = sample_batched['pred_gt'].to(config.device)

        pred_gts = F.interpolate(pred_gts, size=(100, 178), mode='bilinear', align_corners=False)
        pred_gts = pred_gts.detach()

        seg_res, pred = net.forward(seqs, frames)

        d_real_input = F.interpolate(pred_gts, size=(75, 75), mode='bilinear', align_corners=False)
        d_fake_input = F.interpolate(pred.detach(), size=(75, 75), mode='bilinear', align_corners=False)

        netd.eval()
        d_real = netd(d_real_input).squeeze(1)
        d_fake = netd(d_fake_input).squeeze(1)
        netd.train()

        real_label = torch.ones_like(d_real)
        fake_label = torch.zeros_like(d_fake)

        err_d_real = criterion(d_real, real_label)
        err_d_fake = criterion(d_fake, fake_label)

        optimizer.zero_grad()

        if isinstance(seg_res, list):
            seg_loss = 0
            for i, seg_out in enumerate(seg_res):
                weight = 1.0 if i == len(seg_res) - 1 else 0.4
                seg_loss += weight * seg_criterion(seg_out, gts)
            seg_loss = seg_loss / (1.0 + 0.4 * (len(seg_res) - 1))
        else:
            seg_loss = seg_criterion(seg_res, gts)

        seg_loss.backward()
        optimizer.step()

        epoch_loss += seg_loss.item()

        if update_d:
            optimizer_d.zero_grad()
            d_loss = err_d_fake + err_d_real
            d_loss.backward()
            optimizer_d.step()

        lp_loss = None
        if update_g:
            optimizer_g.zero_grad()

            netd.eval()
            d_fake_input = F.interpolate(pred, size=(75, 75), mode='bilinear', align_corners=False)
            d_fake = netd(d_fake_input).squeeze(1)
            netd.train()
            err_g = criterion(d_fake, real_label)

            if pred.shape[-2:] != pred_gts.shape[-2:]:
                pred = F.interpolate(pred, size=pred_gts.shape[-2:], mode='bilinear', align_corners=False)

            lp_loss = lp_function(pred, pred_gts)
            total_loss = lp_loss + config.beta * err_g
            total_loss.backward()
            optimizer_g.step()

            if (err_d_fake.data < config.margin).all() or (err_d_real.data < config.margin).all():
                update_d = False
            if (err_d_fake.data > (1. - config.margin)).all() or (err_d_real.data > (1. - config.margin)).all():
                update_g = False
            if not update_d and not update_g:
                update_d = True
                update_g = True

        if (ii + len(data_loader) * epoch) % 20 == 19 and lp_loss is not None:
            print(f"Iters: [{ii + len(data_loader) * epoch:2d}] time: {timeit.default_timer() - start_time:.4f}, "
                  f"lp_loss: {lp_loss.item():.8f}, G_loss: {err_g.item():.8f}, seg_loss: {seg_loss.item():.8f}")

    return epoch_loss / num_batches


def validate(net, data_loader, seg_criterion, config):
    """Validate the model"""
    net.eval()
    val_loss = 0

    with torch.no_grad():
        for sample in data_loader:
            seqs = sample['images'].to(config.device)
            frames = sample['frame'].to(config.device)
            gts = sample['seg_gt'].to(config.device)

            seg_res, _ = net.forward(seqs, frames)

            if isinstance(seg_res, list):
                seg_loss = seg_criterion(seg_res[-1], gts)
            else:
                seg_loss = seg_criterion(seg_res, gts)

            val_loss += seg_loss.item()

    return val_loss / len(data_loader)


def run_single_round(args, round_num):
    """Run a single training round"""
    print(f"\n{'=' * 80}")
    print(f"STARTING ROUND {round_num + 1}/3")
    print(f"{'=' * 80}\n")

    config = ModelConfig(args, round_num=round_num)

    if torch.cuda.is_available():
        print(f"CUDA available, using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    else:
        print("CUDA not available, using CPU.")

    setup_directories(config)
    train_loader, test_loader = create_data_loaders(config)
    net, netd, criteria, optimizers = create_models_and_optimizers(config)

    log_dir = os.path.join(
        config.save_dir,
        'JointPredSegNet_runs',
        f'round{round_num}_' + datetime.now().strftime('%b%d_%H-%M-%S')
    )
    writer = SummaryWriter(log_dir=log_dir, comment=f'-round{round_num}')

    epoch_losses = []
    val_loss_list = []
    best_val_loss = float('inf')
    best_epoch = -1

    print(f"Starting Training for Round {round_num + 1}...")
    for epoch in range(config.nEpochs):
        train_loss = train_epoch(net, netd, train_loader, optimizers, criteria, config, epoch, writer)
        epoch_losses.append(train_loss)

        val_loss = validate(net, test_loader, criteria[2], config)
        val_loss_list.append(val_loss)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_path = os.path.join(config.save_model_dir, f'{config.model_name}_best.pth')
            torch.save(net.state_dict(), best_model_path)

        print(f"Round {round_num + 1} - Epoch [{epoch + 1}/{config.nEpochs}] - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Save periodic checkpoints
        if (epoch + 1) % config.snapshot == 0:
            checkpoint_path = os.path.join(config.save_model_dir, f'{config.model_name}_epoch{epoch}.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(config.save_model_dir, f'{config.model_name}_final.pth')
    torch.save(net.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

    writer.close()

    # Save losses to file
    results = {
        'round': round_num,
        'train_losses': epoch_losses,
        'val_losses': val_loss_list,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_train_loss': epoch_losses[-1],
        'final_val_loss': val_loss_list[-1]
    }

    results_path = os.path.join(config.results_dir, f'round{round_num}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results


def analyze_multi_round_results(results_list, save_dir):
    """Analyze results across multiple rounds and compute statistics"""
    print(f"\n{'=' * 80}")
    print("STATISTICAL ANALYSIS ACROSS ROUNDS")
    print(f"{'=' * 80}\n")

    # Extract final metrics from each round
    final_train_losses = [r['final_train_loss'] for r in results_list]
    final_val_losses = [r['final_val_loss'] for r in results_list]
    best_val_losses = [r['best_val_loss'] for r in results_list]

    # Compute statistics
    stats_summary = {
        'final_train_loss': {
            'mean': np.mean(final_train_losses),
            'std': np.std(final_train_losses),
            'min': np.min(final_train_losses),
            'max': np.max(final_train_losses),
            'values': final_train_losses
        },
        'final_val_loss': {
            'mean': np.mean(final_val_losses),
            'std': np.std(final_val_losses),
            'min': np.min(final_val_losses),
            'max': np.max(final_val_losses),
            'values': final_val_losses
        },
        'best_val_loss': {
            'mean': np.mean(best_val_losses),
            'std': np.std(best_val_losses),
            'min': np.min(best_val_losses),
            'max': np.max(best_val_losses),
            'values': best_val_losses
        }
    }

    # Print statistics
    print("Final Training Loss:")
    print(
        f"  Mean ± Std: {stats_summary['final_train_loss']['mean']:.6f} ± {stats_summary['final_train_loss']['std']:.6f}")
    print(f"  Range: [{stats_summary['final_train_loss']['min']:.6f}, {stats_summary['final_train_loss']['max']:.6f}]")
    print(f"  Values: {final_train_losses}\n")

    print("Final Validation Loss:")
    print(f"  Mean ± Std: {stats_summary['final_val_loss']['mean']:.6f} ± {stats_summary['final_val_loss']['std']:.6f}")
    print(f"  Range: [{stats_summary['final_val_loss']['min']:.6f}, {stats_summary['final_val_loss']['max']:.6f}]")
    print(f"  Values: {final_val_losses}\n")

    print("Best Validation Loss:")
    print(f"  Mean ± Std: {stats_summary['best_val_loss']['mean']:.6f} ± {stats_summary['best_val_loss']['std']:.6f}")
    print(f"  Range: [{stats_summary['best_val_loss']['min']:.6f}, {stats_summary['best_val_loss']['max']:.6f}]")
    print(f"  Values: {best_val_losses}\n")

    # Save statistics
    stats_path = os.path.join(save_dir, 'multi_round_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=4)

    # Create comparison plots
    plot_multi_round_results(results_list, save_dir)

    return stats_summary


def plot_multi_round_results(results_list, save_dir):
    """Create visualization plots for multi-round results"""

    # Plot 1: Training curves for all rounds
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for i, results in enumerate(results_list):
        epochs = range(1, len(results['train_losses']) + 1)
        plt.plot(epochs, results['train_losses'], label=f'Round {i + 1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Across Rounds')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    for i, results in enumerate(results_list):
        epochs = range(1, len(results['val_losses']) + 1)
        plt.plot(epochs, results['val_losses'], label=f'Round {i + 1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Across Rounds')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    final_train = [r['final_train_loss'] for r in results_list]
    final_val = [r['final_val_loss'] for r in results_list]
    best_val = [r['best_val_loss'] for r in results_list]

    x = np.arange(3)
    width = 0.25

    plt.bar(x - width, final_train, width, label='Final Train Loss', alpha=0.8)
    plt.bar(x, final_val, width, label='Final Val Loss', alpha=0.8)
    plt.bar(x + width, best_val, width, label='Best Val Loss', alpha=0.8)

    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Loss Comparison Across Rounds')
    plt.xticks(x, [f'Round {i + 1}' for i in range(3)])
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multi_round_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Box plots for distribution
    plt.figure(figsize=(10, 6))
    data_to_plot = [
        [r['final_train_loss'] for r in results_list],
        [r['final_val_loss'] for r in results_list],
        [r['best_val_loss'] for r in results_list]
    ]

    plt.boxplot(data_to_plot, labels=['Final Train', 'Final Val', 'Best Val'])
    plt.ylabel('Loss')
    plt.title('Loss Distribution Across 3 Rounds')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, 'loss_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    """Main function to run multiple training rounds"""
    print(f"\n{'=' * 80}")
    print("MULTI-ROUND TRAINING SCRIPT")
    print(f"Running {args.num_rounds} rounds of {100} epochs each")
    print(f"{'=' * 80}\n")

    results_list = []

    for round_num in range(args.num_rounds):
        try:
            results = run_single_round(args, round_num)
            results_list.append(results)

            print(f"\nRound {round_num + 1} completed successfully!")
            print(f"Final Train Loss: {results['final_train_loss']:.6f}")
            print(f"Final Val Loss: {results['final_val_loss']:.6f}")
            print(f"Best Val Loss: {results['best_val_loss']:.6f} (at epoch {results['best_epoch'] + 1})")

        except Exception as e:
            print(f"\nError in round {round_num + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Analyze results across all rounds
    if len(results_list) == args.num_rounds:
        save_dir = os.path.join(Path.save_root_dir(), 'multi_round_results')
        analyze_multi_round_results(results_list, save_dir)
        print(f"\nAll {args.num_rounds} rounds completed successfully!")
        print(f"Results saved in: {save_dir}")
    else:
        print(f"\nWarning: Only {len(results_list)}/{args.num_rounds} rounds completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-round training script for statistical significance")
    parser.add_argument("--frame_nums", type=int, default=4, help="Number of input frames")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of training rounds")
    parser.add_argument(
        "--do_validation",
        action="store_true",
        default=True,
        help="Do validation during training"
    )

    args = parser.parse_args()
    main(args)