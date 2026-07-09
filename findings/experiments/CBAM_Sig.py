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
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json

from network.joint_pred_seg import (
    STCNN,
    FramePredEncoder,
    FramePredDecoder,
    SegEncoder,
    JointSegDecoderCBAM  # Using CBAM decoder
)
from network.googlenet import Inception3

from dataloaders import custom_transforms as tr
from dataloaders import DAVIS_dataloader as davis
from dataloaders import FIRE_dataloader as db
from mypath import Path


class ModelConfig:
    """Configuration class for STCNN-CBAM training"""

    def __init__(self, args, round_num=0, phase='fire'):
        self.gpu_id = 0
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.resume_epoch = 0
        self.phase = phase

        # Adjust epochs based on phase
        if phase == 'davis':
            self.nEpochs = args.davis_epochs
            self.do_validation = False
        else:
            self.nEpochs = args.fire_epochs
            self.do_validation = True

        self.batch_size = 6
        self.snapshot = 10
        self.pred_lr = 1e-8
        self.seg_lr = 1e-4
        self.lr_D = 1e-4
        self.wd = 5e-4
        self.beta = 0.001
        self.margin = 0.3
        self.num_frame = args.frame_nums
        self.round_num = round_num

        # Model naming
        if phase == 'davis':
            self.model_name = f'STCNN_CBAM_DAVIS_pretrain{self.num_frame}_round{round_num}'
        elif phase == 'combined':
            self.model_name = f'STCNN_CBAM_DAVIS_FIRE{self.num_frame}_round{round_num}'
        else:
            self.model_name = f'STCNN_CBAM_FIRE{self.num_frame}_round{round_num}'

        # Paths
        self.save_dir = Path.save_root_dir()
        self.save_model_dir = os.path.join(self.save_dir, self.model_name)
        self.results_dir = os.path.join(self.save_dir, 'stcnn_cbam_multiround', phase)

        # Pretrained paths
        self.pretrained_netd_path = '/home/c43n256/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4/NetD_epoch-99.pth'
        self.pretrained_netg_path = '/home/c43n256/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4/NetG_epoch-99.pth'
        # No pretrained segmentation path - will use ImageNet weights instead

        # For loading Davis pretrained weights
        self.davis_pretrained_path = None
        if phase == 'fire' and args.use_davis_pretrain:
            self.davis_pretrained_path = os.path.join(
                self.save_dir,
                f'STCNN_CBAM_DAVIS_pretrain{self.num_frame}_round{round_num}',
                f'STCNN_CBAM_DAVIS_pretrain{self.num_frame}_round{round_num}_final.pth'
            )


class ModelInitializer:
    """Handles model initialization and weight loading"""

    @staticmethod
    def initialize_netd(netd, model_path, device):
        """Initialize discriminator"""
        try:
            if os.path.exists(model_path):
                print(f"Loading NetD weights from: {model_path}")
                state_dict = torch.load(model_path, map_location=device)
                netd.load_state_dict(state_dict)
            else:
                print("Using torchvision pretrained Inception-v3")
                hub_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
                hub_model.eval()
                pretrained_dict = hub_model.state_dict()
                model_dict = netd.state_dict()
                filtered_dict = {k: v for k, v in pretrained_dict.items()
                                 if k in model_dict and not k.startswith('fc.')}
                model_dict.update(filtered_dict)
                netd.load_state_dict(model_dict)
        except Exception as e:
            print(f"Error loading NetD: {e}")

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
    def initialize_seg_encoder_with_imagenet(seg_enc):
        """Initialize segmentation encoder with ImageNet ResNet101 weights"""
        try:
            from torchvision.models import resnet101, ResNet101_Weights

            print("Initializing SegEncoder with ResNet101 ImageNet weights...")
            pretrained_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = seg_enc.state_dict()

            # Filter out weights that match
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}

            model_dict.update(pretrained_dict)
            seg_enc.load_state_dict(model_dict)

            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} ImageNet weights into SegEncoder")
            return True

        except Exception as e:
            print(f"Error loading ImageNet weights: {e}")
            print("Using random initialization")
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
            encoder_dict = {}

            pattern1 = {k[8:]: v for k, v in pretrained_seg_dict.items() if k.startswith("encoder.")}
            pattern2 = {k[4:]: v for k, v in pretrained_seg_dict.items() if k.startswith("seg.")}
            pattern3 = {k[7:]: v for k, v in pretrained_seg_dict.items() if k.startswith("module.")}
            pattern4 = pretrained_seg_dict

            for pattern_name, pattern in [("encoder.", pattern1), ("seg.", pattern2),
                                          ("module.", pattern3), ("no prefix", pattern4)]:
                matches = sum(1 for k in seg_enc_dict.keys() if k in pattern and 'TempAttention' not in k)
                if matches > len(encoder_dict):
                    encoder_dict = pattern

            compatible_weights = {}
            for k, v in seg_enc_dict.items():
                if 'TempAttention' in k:
                    continue
                if k in encoder_dict and v.shape == encoder_dict[k].shape:
                    compatible_weights[k] = encoder_dict[k]

            if compatible_weights:
                seg_enc_dict.update(compatible_weights)
                seg_enc.load_state_dict(seg_enc_dict, strict=False)
                print(f"Loaded {len(compatible_weights)}/{len(seg_enc_dict)} weights")

            return len(compatible_weights) > 0

        except Exception as e:
            print(f"Error loading segmentation weights: {e}")
            return False

    @staticmethod
    def load_davis_pretrained(net, davis_path, device):
        """Load Davis pretrained weights"""
        if not os.path.exists(davis_path):
            print(f"Davis pretrained weights not found")
            return False

        try:
            print(f"Loading Davis pretrained weights from: {davis_path}")
            checkpoint = torch.load(davis_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(state_dict, strict=False)
            print("Successfully loaded Davis pretrained weights")
            return True
        except Exception as e:
            print(f"Error loading Davis weights: {e}")
            return False


def setup_directories(config):
    """Create necessary directories"""
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.save_model_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)


def create_data_loaders(config, dataset_type='fire'):
    """Create data loaders"""
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
    ])

    if dataset_type == 'davis':
        print("Loading DAVIS dataset...")
        train_set = davis.DAVISDataset(
            inputRes=(256, 256),
            samples_list_file='/home/c43n256/STCNN/data/DAVIS16_samples_list.txt',
            transform=composed_transforms,
            num_frame=config.num_frame
        )
        train_loader = DataLoader(train_set, batch_size=config.batch_size,
                                  shuffle=True, num_workers=4)
        val_loader = None

    else:  # fire
        print("Loading FIRE dataset...")
        train_set = db.FIREDataset(
            inputRes=(256, 256),
            mode="train",
            num_frame=config.num_frame
        )
        train_loader = DataLoader(train_set, batch_size=config.batch_size,
                                  shuffle=True, num_workers=4)

        if config.do_validation:
            val_set = db.FIREDataset(
                inputRes=(256, 256),
                mode="test",
                num_frame=config.num_frame
            )
            val_loader = DataLoader(val_set, batch_size=1,
                                    shuffle=False, num_workers=4)
        else:
            val_loader = None

    return train_loader, val_loader


def create_models_and_optimizers(config):
    """Create models with CBAM decoder"""
    netd = Inception3(num_classes=1, aux_logits=False, transform_input=True)
    pred_enc = FramePredEncoder(frame_nums=config.num_frame)
    pred_dec = FramePredDecoder()
    seg_enc = SegEncoder()
    seg_dec = JointSegDecoderCBAM()  # Using CBAM decoder

    initializer = ModelInitializer()

    # Initialize discriminator and prediction branch
    initializer.initialize_netd(netd, config.pretrained_netd_path, config.device)
    initializer.load_prediction_weights(pred_enc, pred_dec, config.pretrained_netg_path, config.device)

    # Create STCNN
    net = STCNN(pred_enc, pred_dec, seg_enc, seg_dec)

    # Load appropriate weights based on phase
    if config.phase == 'davis':
        # Initialize with ImageNet weights for Davis pretraining
        initializer.initialize_seg_encoder_with_imagenet(seg_enc)
        print("Initialized SegEncoder with ImageNet weights for Davis pretraining")
    elif config.phase == 'fire' and config.davis_pretrained_path:
        # Load Davis pretrained weights
        if initializer.load_davis_pretrained(net, config.davis_pretrained_path, config.device):
            print("Loaded Davis pretrained CBAM model for FIRE training")
        else:
            print("Davis weights not found, initializing with ImageNet")
            initializer.initialize_seg_encoder_with_imagenet(seg_enc)
    else:
        # FIRE training without Davis pretraining - use ImageNet
        initializer.initialize_seg_encoder_with_imagenet(seg_enc)
        print("Initialized SegEncoder with ImageNet weights for FIRE training")

    net.to(config.device)
    netd.to(config.device)

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss functions
    lp_function = nn.MSELoss().to(config.device)
    criterion = nn.BCELoss().to(config.device)
    seg_criterion = nn.BCEWithLogitsLoss().to(config.device)

    # Optimizers
    optimizer = optim.SGD([
        {'params': net.seg_decoder.parameters(), 'lr': config.seg_lr},
        {'params': net.seg_encoder.parameters(), 'lr': config.seg_lr},
    ], weight_decay=config.wd, momentum=0.9)

    optimizer_g = optim.Adam([
        {'params': net.pred_encoder.parameters(), 'lr': config.pred_lr},
        {'params': net.pred_decoder.parameters(), 'lr': config.pred_lr},
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
            print(f"Iters: [{ii + len(data_loader) * epoch:2d}] "
                  f"lp_loss: {lp_loss.item():.8f}, seg_loss: {seg_loss.item():.8f}")

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


def run_single_round(args, round_num, phase='fire'):
    """Run a single training round"""
    print(f"\n{'=' * 80}")
    print(f"STARTING ROUND {round_num + 1}/3 - PHASE: {phase.upper()} - MODEL: STCNN+CBAM")
    print(f"{'=' * 80}\n")

    config = ModelConfig(args, round_num=round_num, phase=phase)

    if torch.cuda.is_available():
        print(f"Using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")

    setup_directories(config)

    dataset_type = 'davis' if phase == 'davis' else 'fire'
    train_loader, val_loader = create_data_loaders(config, dataset_type)

    net, netd, criteria, optimizers = create_models_and_optimizers(config)

    log_dir = os.path.join(
        config.save_dir,
        'STCNN_CBAM_runs',
        f'{phase}_round{round_num}_' + datetime.now().strftime('%b%d_%H-%M-%S')
    )
    writer = SummaryWriter(log_dir=log_dir, comment=f'-{phase}-cbam-round{round_num}')

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = -1

    print(f"Starting {phase.upper()} training with CBAM...")
    for epoch in range(config.nEpochs):
        train_loss = train_epoch(net, netd, train_loader, optimizers, criteria, config, epoch, writer)
        train_losses.append(train_loss)

        val_loss = 0
        if config.do_validation and val_loader:
            val_loss = validate(net, val_loader, criteria[2], config)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_path = os.path.join(config.save_model_dir, f'{config.model_name}_best.pth')
                torch.save(net.state_dict(), best_path)

        print(f"{phase.upper()} Round {round_num + 1} - Epoch [{epoch + 1}/{config.nEpochs}] - "
              f"Train: {train_loss:.6f}" + (f", Val: {val_loss:.6f}" if config.do_validation else ""))

        writer.add_scalar('Loss/train', train_loss, epoch)
        if config.do_validation:
            writer.add_scalar('Loss/val', val_loss, epoch)

    # Save final model
    final_path = os.path.join(config.save_model_dir, f'{config.model_name}_final.pth')
    torch.save(net.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

    writer.close()

    results = {
        'round': round_num,
        'phase': phase,
        'model': 'STCNN_CBAM',
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

    plot_training_curves(train_losses, val_losses if config.do_validation else None,
                         os.path.join(config.save_model_dir, 'training_curve.png'),
                         phase, round_num)

    return results


def plot_training_curves(train_losses, val_losses, save_path, phase, round_num):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'STCNN-CBAM - {phase.upper()} Round {round_num + 1}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def analyze_multi_round_results(results_list, save_dir, phase):
    """Analyze results across rounds"""
    print(f"\n{'=' * 80}")
    print(f"STATISTICAL ANALYSIS - STCNN-CBAM {phase.upper()}")
    print(f"{'=' * 80}\n")

    final_train_losses = [r['final_train_loss'] for r in results_list]

    stats_summary = {
        'phase': phase,
        'model': 'STCNN_CBAM',
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
            'values': final_val_losses
        }

        stats_summary['best_val_loss'] = {
            'mean': np.mean(best_val_losses),
            'std': np.std(best_val_losses),
            'values': best_val_losses
        }

    print("Final Training Loss:")
    print(f"  Mean ± Std: {stats_summary['final_train_loss']['mean']:.6f} ± "
          f"{stats_summary['final_train_loss']['std']:.6f}")
    print(f"  Values: {final_train_losses}\n")

    if 'final_val_loss' in stats_summary:
        print("Final Validation Loss:")
        print(f"  Mean ± Std: {stats_summary['final_val_loss']['mean']:.6f} ± "
              f"{stats_summary['final_val_loss']['std']:.6f}\n")

        print("Best Validation Loss:")
        print(f"  Mean ± Std: {stats_summary['best_val_loss']['mean']:.6f} ± "
              f"{stats_summary['best_val_loss']['std']:.6f}\n")

    stats_path = os.path.join(save_dir, f'{phase}_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=4)

    return stats_summary


def main(args):
    """Main function"""
    print(f"\n{'=' * 80}")
    print("STCNN-CBAM MULTI-ROUND TRAINING")
    print(f"Training mode: {args.training_mode}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"{'=' * 80}\n")

    save_dir = os.path.join(Path.save_root_dir(), 'stcnn_cbam_multiround')

    if args.training_mode == 'davis_only':
        results_list = []
        for round_num in range(args.num_rounds):
            try:
                results = run_single_round(args, round_num, phase='davis')
                results_list.append(results)
                print(f"\nDavis Round {round_num + 1} completed! Loss: {results['final_train_loss']:.6f}")
            except Exception as e:
                print(f"Error in Davis round {round_num + 1}: {e}")
                import traceback
                traceback.print_exc()

        if len(results_list) == args.num_rounds:
            analyze_multi_round_results(results_list, os.path.join(save_dir, 'davis'), 'davis')

    elif args.training_mode == 'fire_only':
        results_list = []
        for round_num in range(args.num_rounds):
            try:
                results = run_single_round(args, round_num, phase='fire')
                results_list.append(results)
                print(f"\nFIRE Round {round_num + 1} completed!")
                print(f"Train: {results['final_train_loss']:.6f}, Val: {results['final_val_loss']:.6f}")
            except Exception as e:
                print(f"Error in FIRE round {round_num + 1}: {e}")
                import traceback
                traceback.print_exc()

        if len(results_list) == args.num_rounds:
            analyze_multi_round_results(results_list, os.path.join(save_dir, 'fire'), 'fire')

    elif args.training_mode == 'combined':
        davis_results = []
        fire_results = []

        for round_num in range(args.num_rounds):
            print(f"\n{'=' * 80}")
            print(f"ROUND {round_num + 1}/{args.num_rounds}")
            print(f"{'=' * 80}\n")

            try:
                print("Phase 1: Davis Pretraining")
                davis_result = run_single_round(args, round_num, phase='davis')
                davis_results.append(davis_result)
                print(f"Davis completed - Loss: {davis_result['final_train_loss']:.6f}")
            except Exception as e:
                print(f"Error in Davis: {e}")
                import traceback
                traceback.print_exc()
                continue

            try:
                print("\nPhase 2: FIRE Training")
                fire_result = run_single_round(args, round_num, phase='fire')
                fire_results.append(fire_result)
                print(f"FIRE completed - Train: {fire_result['final_train_loss']:.6f}, "
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
        description="Multi-round STCNN-CBAM training with Davis pretraining")

    parser.add_argument("--frame_nums", type=int, default=4,
                        help="Number of input frames")

    parser.add_argument("--num_rounds", type=int, default=3,
                        help="Number of training rounds")

    parser.add_argument("--training_mode", type=str, default="fire_only",
                        choices=["davis_only", "fire_only", "combined"],
                        help="Training mode")

    parser.add_argument("--davis_epochs", type=int, default=50,
                        help="Epochs for Davis pretraining")

    parser.add_argument("--fire_epochs", type=int, default=100,
                        help="Epochs for FIRE training")

    parser.add_argument("--use_davis_pretrain", action="store_true",
                        help="Use Davis pretrained weights")

    args = parser.parse_args()

    if args.training_mode == 'combined':
        args.use_davis_pretrain = True

    main(args)