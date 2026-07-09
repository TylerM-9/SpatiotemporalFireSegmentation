"""
Multi-round STCNN training (DAVIS pretrain + FIRE finetune) using JointSegDecoderNoPPM

This is based directly on the multi-round STCNN training file you pasted earlier
(the one with ModelConfig / ModelInitializer / run_single_round / training_mode).

ONLY change vs your original multi-round STCNN file:
  - Use JointSegDecoderNoPPM as the segmentation decoder (instead of JointSegDecoder)
  - Model naming includes "NoPPM" so it doesn't overwrite your other runs

Phases / modes:
  --training_mode davis_only   : DAVIS pretrain only (no validation)
  --training_mode fire_only    : FIRE only (no DAVIS pretrain)
  --training_mode combined     : DAVIS pretrain first, then FIRE using DAVIS-pretrained weights

Saves:
  *_best.pth  (FIRE only, based on val loss)
  *_epoch{E}.pth
  *_final.pth
  multi_round_results JSON + comparison plots
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
from datetime import datetime
import timeit
import json

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from dataloaders import custom_transforms as tr
from dataloaders import DAVIS_dataloader as davis
from dataloaders import FIRE_dataloader as db
from mypath import Path

# Discriminator
from network.googlenet import Inception3

# STCNN components
from network.joint_pred_seg import (
    STCNN,
    FramePredEncoder,
    FramePredDecoder,
    SegEncoder,
    JointSegDecoderNoPPM,   # <<< IMPORTANT: NoPPM decoder
)


# --------------------------
# Config
# --------------------------
class ModelConfig:
    """Configuration class to hold all model parameters"""

    def __init__(self, args, round_num=0, phase='fire'):
        self.gpu_id = 0
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.resume_epoch = 0
        self.phase = phase  # 'davis' or 'fire' (combined uses both sequentially)

        # Epochs / validation
        if phase == 'davis':
            self.nEpochs = args.davis_epochs
            self.do_validation = False
        else:
            self.nEpochs = args.fire_epochs
            self.do_validation = True

        # Training hyperparams (copied from your earlier multi-round STCNN file)
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

        # ---------
        # Model naming (include NoPPM to avoid overwriting other runs)
        # ---------
        if phase == 'davis':
            self.model_name = f'STCNN_NoPPM_DAVIS_pretrain{self.num_frame}_round{round_num}'
        else:
            self.model_name = f'STCNN_NoPPM_FIRE{self.num_frame}_round{round_num}'

        # Paths
        self.save_dir = Path.save_root_dir()
        self.save_model_dir = os.path.join(self.save_dir, self.model_name)
        self.results_dir = os.path.join(self.save_dir, 'multi_round_results', f'NoPPM_{phase}')

        # Pretrained paths (same as your file)
        self.pretrained_netd_path = '/home/c43n256/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4/NetD_epoch-99.pth'
        self.pretrained_netg_path = '/home/c43n256/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4/NetG_epoch-99.pth'
        self.pretrained_seg_path = '/home/c43n256/STCNN/output/Seg_Branch_NEW_RUN/Seg_Branch_NEW_RUN_epoch-11999.pth'

        # For FIRE phase when using davis pretrain (combined mode)
        self.davis_pretrained_path = None
        if phase == 'fire' and args.use_davis_pretrain:
            self.davis_pretrained_path = os.path.join(
                self.save_dir,
                f'STCNN_NoPPM_DAVIS_pretrain{self.num_frame}_round{round_num}',
                f'STCNN_NoPPM_DAVIS_pretrain{self.num_frame}_round{round_num}_final.pth'
            )


# --------------------------
# Initializer (copied from your multi-round file)
# --------------------------
class ModelInitializer:
    """Handles model initialization and weight loading"""

    @staticmethod
    def initialize_netd(netd, model_path, device):
        """Initialize discriminator with pretrained weights (if available), else fallback"""
        try:
            if os.path.exists(model_path):
                print(f"Loading NetD weights from: {model_path}")
                state_dict = torch.load(model_path, map_location=device)
                netd.load_state_dict(state_dict)
            else:
                print("Saved NetD weights not found, using torchvision InceptionV3 pretrained weights (hub)")
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
        """Load prediction encoder and decoder weights from NetG checkpoint"""
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
        """Load segmentation encoder weights from your SegBranch checkpoint"""
        if not os.path.exists(seg_path):
            print(f"Segmentation weights not found at: {seg_path}")
            return False

        try:
            print(f"Loading segmentation weights from: {seg_path}")
            checkpoint = torch.load(seg_path, map_location=device)
            pretrained_seg_dict = checkpoint['state_dict'] if (isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint

            seg_enc_dict = seg_enc.state_dict()

            pattern1 = {k[8:]: v for k, v in pretrained_seg_dict.items() if k.startswith("encoder.")}
            pattern2 = {k[4:]: v for k, v in pretrained_seg_dict.items() if k.startswith("seg.")}
            pattern3 = {k[7:]: v for k, v in pretrained_seg_dict.items() if k.startswith("module.")}
            pattern4 = pretrained_seg_dict

            encoder_dict = {}
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
                if k in encoder_dict and v.shape == encoder_dict[k].shape:
                    compatible_weights[k] = encoder_dict[k]

            if compatible_weights:
                seg_enc_dict.update(compatible_weights)
                seg_enc.load_state_dict(seg_enc_dict, strict=False)
                print(f"Successfully loaded {len(compatible_weights)}/{len(seg_enc_dict)} weights into SegEncoder")
                return True

            print("No compatible weights found for SegEncoder")
            return False

        except Exception as e:
            print(f"Error loading segmentation weights: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def load_davis_pretrained(net, davis_path, device):
        """Load full STCNN weights from Davis pretraining checkpoint"""
        if not os.path.exists(davis_path):
            print(f"Davis pretrained weights not found at: {davis_path}")
            return False

        try:
            print(f"Loading Davis pretrained weights from: {davis_path}")
            checkpoint = torch.load(davis_path, map_location=device)
            state_dict = checkpoint['state_dict'] if (isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint

            missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")

            print("Successfully loaded Davis pretrained weights")
            return True
        except Exception as e:
            print(f"Error loading Davis pretrained weights: {e}")
            return False


# --------------------------
# Utilities
# --------------------------
def setup_directories(config):
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.save_model_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)


def create_data_loaders(config, dataset_type='fire'):
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
    ])

    if dataset_type == 'davis':
        print("Loading DAVIS dataset for pretraining...")
        train_set = davis.DAVISDataset(
            inputRes=(256, 256),
            samples_list_file=os.path.join('/home/c43n256/STCNN/data/DAVIS16_samples_list.txt'),
            transform=composed_transforms,
            num_frame=config.num_frame
        )
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
        test_loader = None

    else:
        print("Loading FIRE dataset for training...")
        train_set = db.FIREDataset(inputRes=(256, 256), mode="train", num_frame=config.num_frame)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)

        if config.do_validation:
            test_set = db.FIREDataset(inputRes=(256, 256), mode="test", num_frame=config.num_frame)
            test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)
        else:
            test_loader = None

    return train_loader, test_loader


def create_models_and_optimizers(config):
    # Discriminator + STCNN branches
    netd = Inception3(num_classes=1, aux_logits=False, transform_input=True)
    pred_enc = FramePredEncoder(frame_nums=config.num_frame)
    pred_dec = FramePredDecoder()
    seg_enc = SegEncoder()

    # <<< IMPORTANT: NoPPM decoder here
    seg_dec = JointSegDecoderNoPPM()

    initializer = ModelInitializer()

    # Init NetD + prediction weights always
    initializer.initialize_netd(netd, config.pretrained_netd_path, config.device)
    initializer.load_prediction_weights(pred_enc, pred_dec, config.pretrained_netg_path, config.device)

    # Build STCNN
    net = STCNN(pred_enc, pred_dec, seg_enc, seg_dec)

    # Load weights depending on phase
    if config.phase == 'davis':
        initializer.load_segmentation_weights(seg_enc, config.pretrained_seg_path, config.device)
        print("Initialized for DAVIS pretraining (NoPPM)")

    elif config.phase == 'fire' and config.davis_pretrained_path:
        if initializer.load_davis_pretrained(net, config.davis_pretrained_path, config.device):
            print("Loaded DAVIS-pretrained STCNN (NoPPM) for FIRE training")
        else:
            print("Could not load DAVIS-pretrained model; falling back to base SegEncoder weights")
            initializer.load_segmentation_weights(seg_enc, config.pretrained_seg_path, config.device)
    else:
        initializer.load_segmentation_weights(seg_enc, config.pretrained_seg_path, config.device)
        print("Initialized for FIRE training (NoPPM, no DAVIS pretraining)")

    net.to(config.device)
    netd.to(config.device)

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Losses
    lp_function = nn.MSELoss().to(config.device)
    criterion = nn.BCELoss().to(config.device)
    seg_criterion = nn.BCEWithLogitsLoss().to(config.device)

    # Optimizers (same as your file)
    optimizer = optim.SGD([
        {'params': [p for _, p in net.seg_decoder.named_parameters()], 'lr': config.seg_lr},
        {'params': [p for _, p in net.seg_encoder.named_parameters()], 'lr': config.seg_lr},
    ], weight_decay=config.wd, momentum=0.9)

    optimizer_g = optim.Adam([
        {'params': [p for _, p in net.pred_encoder.named_parameters()], 'lr': config.pred_lr},
        {'params': [p for _, p in net.pred_decoder.named_parameters()], 'lr': config.pred_lr},
    ], lr=config.pred_lr, weight_decay=config.wd)

    optimizer_d = optim.Adam(netd.parameters(), lr=config.lr_D, weight_decay=config.wd)

    return net, netd, (lp_function, criterion, seg_criterion), (optimizer, optimizer_g, optimizer_d)


# --------------------------
# Train / Validate (copied logic)
# --------------------------
def train_epoch(net, netd, data_loader, optimizers, criteria, config, epoch, writer):
    optimizer, optimizer_g, optimizer_d = optimizers
    lp_function, criterion, seg_criterion = criteria

    net.train()
    netd.train()

    epoch_loss = 0.0
    num_batches = len(data_loader)
    start_time = timeit.default_timer()

    update_d = True
    update_g = False  # keep as your script (only seg updates by default)

    for ii, sample_batched in enumerate(data_loader):
        seqs = sample_batched['images'].to(config.device).requires_grad_()
        frames = sample_batched['frame'].to(config.device).requires_grad_()
        gts = sample_batched['seg_gt'].to(config.device)
        pred_gts = sample_batched['pred_gt'].to(config.device)

        pred_gts = F.interpolate(pred_gts, size=(100, 178), mode='bilinear', align_corners=False)
        pred_gts = pred_gts.detach()

        seg_res, pred = net.forward(seqs, frames)

        # Discriminator inputs
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

        # ---- Seg update ----
        optimizer.zero_grad()

        if isinstance(seg_res, list):
            seg_loss = 0.0
            for i, seg_out in enumerate(seg_res):
                weight = 1.0 if i == len(seg_res) - 1 else 0.4
                seg_loss += weight * seg_criterion(seg_out, gts)
            seg_loss = seg_loss / (1.0 + 0.4 * (len(seg_res) - 1))
        else:
            seg_loss = seg_criterion(seg_res, gts)

        seg_loss.backward()
        optimizer.step()
        epoch_loss += float(seg_loss.item())

        # ---- D update ----
        if update_d:
            optimizer_d.zero_grad()
            d_loss = err_d_fake + err_d_real
            d_loss.backward()
            optimizer_d.step()

        # ---- G update (optional, kept same structure) ----
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

    return epoch_loss / max(1, num_batches)


def validate(net, data_loader, seg_criterion, config):
    net.eval()
    val_loss = 0.0

    with torch.no_grad():
        for sample in data_loader:
            seqs = sample['images'].to(config.device)
            frames = sample['frame'].to(config.device)
            gts = sample['seg_gt'].to(config.device)

            seg_res, _ = net.forward(seqs, frames)
            seg_logits = seg_res[-1] if isinstance(seg_res, list) else seg_res
            seg_loss = seg_criterion(seg_logits, gts)
            val_loss += float(seg_loss.item())

    return val_loss / max(1, len(data_loader))


# --------------------------
# Round runner + analysis (copied structure)
# --------------------------
def run_single_round(args, round_num, phase='fire'):
    print(f"\n{'=' * 80}")
    print(f"STARTING ROUND {round_num + 1}/{args.num_rounds} - PHASE: {phase.upper()} (NoPPM)")
    print(f"{'=' * 80}\n")

    config = ModelConfig(args, round_num=round_num, phase=phase)

    if torch.cuda.is_available():
        print(f"CUDA available, using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    else:
        print("CUDA not available, using CPU.")

    setup_directories(config)

    dataset_type = 'davis' if phase == 'davis' else 'fire'
    train_loader, test_loader = create_data_loaders(config, dataset_type)

    net, netd, criteria, optimizers = create_models_and_optimizers(config)

    log_dir = os.path.join(
        config.save_dir,
        'JointPredSegNet_runs_NoPPM',
        f'{phase}_round{round_num}_' + datetime.now().strftime('%b%d_%H-%M-%S')
    )
    writer = SummaryWriter(log_dir=log_dir, comment=f'-NoPPM-{phase}-round{round_num}')

    epoch_losses = []
    val_loss_list = []
    best_val_loss = float('inf')
    best_epoch = -1

    print(f"Starting {phase.upper()} Training for Round {round_num + 1}...")
    for epoch in range(config.nEpochs):
        train_loss = train_epoch(net, netd, train_loader, optimizers, criteria, config, epoch, writer)
        epoch_losses.append(train_loss)

        val_loss = 0.0
        if config.do_validation and test_loader is not None:
            val_loss = validate(net, test_loader, criteria[2], config)
            val_loss_list.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_path = os.path.join(config.save_model_dir, f'{config.model_name}_best.pth')
                torch.save(net.state_dict(), best_model_path)

        print(f"{phase.upper()} Round {round_num + 1} - Epoch [{epoch + 1}/{config.nEpochs}] - "
              f"Train Loss: {train_loss:.6f}" + (f", Val Loss: {val_loss:.6f}" if config.do_validation else ""))

        writer.add_scalar('Loss/train', train_loss, epoch)
        if config.do_validation:
            writer.add_scalar('Loss/val', val_loss, epoch)

        if (epoch + 1) % config.snapshot == 0:
            checkpoint_path = os.path.join(config.save_model_dir, f'{config.model_name}_epoch{epoch}.pth')
            torch.save(net.state_dict(), checkpoint_path)

    final_model_path = os.path.join(config.save_model_dir, f'{config.model_name}_final.pth')
    torch.save(net.state_dict(), final_model_path)

    writer.close()

    results = {
        'round': round_num,
        'phase': phase,
        'train_losses': epoch_losses,
        'val_losses': val_loss_list if config.do_validation else [],
        'best_val_loss': best_val_loss if config.do_validation else None,
        'best_epoch': best_epoch if config.do_validation else None,
        'final_train_loss': epoch_losses[-1] if epoch_losses else None,
        'final_val_loss': val_loss_list[-1] if val_loss_list else None
    }

    results_path = os.path.join(config.results_dir, f'round{round_num}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results


def analyze_multi_round_results(results_list, save_dir, phase):
    print(f"\n{'=' * 80}")
    print(f"STATISTICAL ANALYSIS - NoPPM {phase.upper()} PHASE")
    print(f"{'=' * 80}\n")

    final_train_losses = [r['final_train_loss'] for r in results_list if r['final_train_loss'] is not None]

    stats_summary = {
        'phase': phase,
        'variant': 'NoPPM',
        'final_train_loss': {
            'mean': float(np.mean(final_train_losses)),
            'std': float(np.std(final_train_losses)),
            'min': float(np.min(final_train_losses)),
            'max': float(np.max(final_train_losses)),
            'values': final_train_losses
        }
    }

    if results_list and results_list[0].get('final_val_loss', None) is not None:
        final_val_losses = [r['final_val_loss'] for r in results_list if r['final_val_loss'] is not None]
        best_val_losses = [r['best_val_loss'] for r in results_list if r['best_val_loss'] is not None]

        stats_summary['final_val_loss'] = {
            'mean': float(np.mean(final_val_losses)),
            'std': float(np.std(final_val_losses)),
            'min': float(np.min(final_val_losses)),
            'max': float(np.max(final_val_losses)),
            'values': final_val_losses
        }
        stats_summary['best_val_loss'] = {
            'mean': float(np.mean(best_val_losses)),
            'std': float(np.std(best_val_losses)),
            'min': float(np.min(best_val_losses)),
            'max': float(np.max(best_val_losses)),
            'values': best_val_losses
        }

    os.makedirs(save_dir, exist_ok=True)
    stats_path = os.path.join(save_dir, f'NoPPM_{phase}_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=4)

    plot_multi_round_results(results_list, save_dir, phase)
    return stats_summary


def plot_multi_round_results(results_list, save_dir, phase):
    has_val = bool(results_list) and results_list[0].get('final_val_loss', None) is not None

    plt.figure(figsize=(15, 5) if has_val else (10, 5))
    if has_val:
        plt.subplot(1, 2, 1)

    for i, results in enumerate(results_list):
        epochs = range(1, len(results['train_losses']) + 1)
        plt.plot(epochs, results['train_losses'], label=f'Round {i + 1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss - NoPPM {phase.upper()}')
    plt.legend()
    plt.grid(True)

    if has_val:
        plt.subplot(1, 2, 2)
        for i, results in enumerate(results_list):
            epochs = range(1, len(results['val_losses']) + 1)
            plt.plot(epochs, results['val_losses'], label=f'Round {i + 1}', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title(f'Validation Loss - NoPPM {phase.upper()}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'NoPPM_{phase}_comparison.png'), dpi=300)
    plt.close()


# --------------------------
# Main
# --------------------------
def main(args):
    print(f"\n{'=' * 80}")
    print("STCNN MULTI-ROUND TRAINING (NoPPM Decoder)")
    print(f"Training mode: {args.training_mode}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Frames: {args.frame_nums}")
    print(f"{'=' * 80}\n")

    base_save_dir = os.path.join(Path.save_root_dir(), 'multi_round_results')

    if args.training_mode == 'davis_only':
        results_list = []
        for round_num in range(args.num_rounds):
            try:
                results = run_single_round(args, round_num, phase='davis')
                results_list.append(results)
                print(f"\nDAVIS Round {round_num + 1} completed! Final Train Loss: {results['final_train_loss']:.6f}")
            except Exception as e:
                print(f"\nError in DAVIS round {round_num + 1}: {str(e)}")
                import traceback
                traceback.print_exc()

        if len(results_list) == args.num_rounds:
            analyze_multi_round_results(results_list, os.path.join(base_save_dir, 'NoPPM_davis'), 'davis')

    elif args.training_mode == 'fire_only':
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
            analyze_multi_round_results(results_list, os.path.join(base_save_dir, 'NoPPM_fire'), 'fire')

    elif args.training_mode == 'combined':
        # Round-by-round: DAVIS then FIRE (using davis pretrained)
        davis_results = []
        fire_results = []

        for round_num in range(args.num_rounds):
            print(f"\n{'=' * 80}")
            print(f"ROUND {round_num + 1}/{args.num_rounds} (NoPPM)")
            print(f"{'=' * 80}\n")

            # Phase 1: DAVIS
            try:
                print("Phase 1: DAVIS pretraining")
                dr = run_single_round(args, round_num, phase='davis')
                davis_results.append(dr)
                print(f"DAVIS done. Final Train Loss: {dr['final_train_loss']:.6f}")
            except Exception as e:
                print(f"Error in DAVIS round {round_num + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

            # Phase 2: FIRE (load davis pretrained)
            try:
                print("Phase 2: FIRE training (using DAVIS pretrained weights)")
                fr = run_single_round(args, round_num, phase='fire')
                fire_results.append(fr)
                print(f"FIRE done. Final Train Loss: {fr['final_train_loss']:.6f}, Final Val Loss: {fr['final_val_loss']:.6f}")
            except Exception as e:
                print(f"Error in FIRE round {round_num + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        if len(davis_results) == args.num_rounds:
            analyze_multi_round_results(davis_results, os.path.join(base_save_dir, 'NoPPM_combined_davis'), 'davis')
        if len(fire_results) == args.num_rounds:
            analyze_multi_round_results(fire_results, os.path.join(base_save_dir, 'NoPPM_combined_fire'), 'fire')

        print("\nCombined training completed.")
        print(f"DAVIS rounds: {len(davis_results)}/{args.num_rounds}")
        print(f"FIRE rounds: {len(fire_results)}/{args.num_rounds}")

    else:
        raise ValueError(f"Unknown training mode: {args.training_mode}")


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-round STCNN training with NoPPM JointSegDecoder")

    parser.add_argument("--frame_nums", type=int, default=4, help="Number of input frames")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of training rounds")

    parser.add_argument("--training_mode", type=str, default="combined",
                        choices=["davis_only", "fire_only", "combined"],
                        help="davis_only (pretrain), fire_only (train only), combined (davis then fire)")

    parser.add_argument("--davis_epochs", type=int, default=50, help="Epochs for DAVIS pretraining")
    parser.add_argument("--fire_epochs", type=int, default=100, help="Epochs for FIRE training")

    parser.add_argument("--use_davis_pretrain", action="store_true",
                        help="Use DAVIS pretrained weights for FIRE training (auto-enabled in combined)")

    args = parser.parse_args()

    if args.training_mode == 'combined':
        args.use_davis_pretrain = True

    main(args)
