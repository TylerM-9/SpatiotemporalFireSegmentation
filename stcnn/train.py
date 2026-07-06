"""
SIMPLIFIED ST-UNET TRAINING SCRIPT
Uses UNet encoder/decoder with temporal attention from network/UNET_ST.py
"""

from __future__ import absolute_import, division, print_function

import argparse
from ast import parse
import os
from datetime import datetime
import socket
import timeit

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Use the clean ST-UNet architecture
from network.UNET_ST import create_stunet_with_attention
from network.STUNet3Plus import create_unet3plus
from network.joint_pred_seg import FramePredDecoder, FramePredEncoder
from network.googlenet import Inception3

from dataloaders import custom_transforms as tr
from dataloaders import DAVIS_dataloader as davis
from dataloaders import FIRE_dataloader as db
from mypath import Path

# GPU
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")


def main(args):
    # Select which GPU, -1 if CPU
    if torch.cuda.is_available():
        print(f"CUDA available, using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("CUDA not available, using CPU.")

    # ------------------------------
    # Settings
    # ------------------------------
    resume_epoch = args.resume_epoch
    nEpochs = args.epochs
    batch_size = 6
    snapshot = 5

    # LRs / reg
    pred_lr = 1e-8       # Temporal branch stays frozen
    seg_lr = 1e-4        # Main LR for segmentation branch
    lr_D = 1e-4
    wd = 5e-4
    beta = 0.001
    margin = 0.3

    updateD = True
    updateG = False

    num_frame = args.frame_nums
    dataset_type = args.dataset  # 'davis' or 'fire'
    model = args.model  # 'stunet' or 'stunet3plus'

    if (model == "stunet"):
        modelName = f'STUNET_UNET_DAVIS_{dataset_type.upper()}{num_frame}'
    else:
        modelName = f'STUNET3PLUS_DAVIS_{dataset_type.upper()}{num_frame}'

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    save_model_dir = os.path.join(save_dir, modelName)
    os.makedirs(save_model_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training Configuration:")
    print(f"  Dataset: {dataset_type.upper()}")
    print(f"  Model: {model}")
    print(f"  Frames: {num_frame}")
    print(f"  Epochs: {nEpochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Resume from epoch: {resume_epoch}")
    print(f"{'=' * 60}\n")

    # ------------------------------
    # Networks
    # ------------------------------
    # Discriminator (for temporal NetG supervision)
    netD = Inception3(num_classes=1, aux_logits=False, transform_input=True)
    initialize_netD(netD, os.path.join(
        '/home/c43n256/REU2026/FramePredModels/frames_nums_4',
        'NetD_epoch-99.pth'))

    # Temporal prediction branch (pretrained)
    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()

    print("Loading weights from pretrained NetG")
    pretrained_netG_dict = torch.load(
        os.path.join(
            '/home/c43n256/REU2026/FramePredModels/frames_nums_4',
            'NetG_epoch-99.pth'),
        map_location=device
    )

    # Load pred_enc weights
    model_dict = pred_enc.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    pred_enc.load_state_dict(model_dict)

    # Load pred_dec weights
    model_dict = pred_dec.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    pred_dec.load_state_dict(model_dict)
    print("Temporal branch loaded successfully")

    print("Building Spatio-Temporal Model...")
    
    # Select model based on flag
    if model == "stunet3plus":
        print("Initializing STUNet3Plus Architecture (UNet3+ Full-Scale Skip Connections)")
        net = create_unet3plus(pred_enc=pred_enc, pred_dec=pred_dec, num_frame=num_frame, n_classes=1)
    else:
        print("Initializing Baseline ST-UNet Architecture")
        net = create_stunet_with_attention(pred_enc=pred_enc, pred_dec=pred_dec, num_frame=num_frame, n_classes=1)

    # # ST-UNet (UNet encoder/decoder with attention)
    # print("Creating ST-UNet (UNet encoder/decoder) with temporal attention")
    # net = create_stunet_with_attention(
    #     pred_enc=pred_enc,
    #     pred_dec=pred_dec,
    #     num_frame=num_frame,
    #     n_classes=1
    # )

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Optional: load pretrained segmentation weights
    if args.pretrained_seg is not None:
        print(f"Loading pretrained segmentation weights from: {args.pretrained_seg}")
        if os.path.exists(args.pretrained_seg):
            checkpoint = torch.load(args.pretrained_seg, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                net.load_state_dict(checkpoint['state_dict'], strict=False)
                print("Successfully loaded pretrained segmentation checkpoint")
            else:
                net.load_state_dict(checkpoint, strict=False)
                print("Successfully loaded pretrained segmentation weights")
        else:
            print(f"Warning: Pretrained checkpoint not found at {args.pretrained_seg}")
            print("Continuing without pretrained segmentation weights")
    elif resume_epoch > 0:
        # Try to load from standard resume path
        if model == "stunet":
            resume_path = os.path.join("/home/c43n256/REU2026/SpatiotemporalFireSegmentation/stcnn/output/STUNET_UNET_DAVIS4/STUNET_UNET_DAVIS4-199.pth")
        else: 
            resume_path = os.path.join("/home/c43n256/REU2026/SpatiotemporalFireSegmentation/stcnn/output/STUNET3PLUS_DAVIS4/STUNET3PLUS_DAVIS4-199.pth")

        if os.path.exists(resume_path):
            print(f"Resuming from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                net.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                net.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded resume checkpoint")
        else:
            print(f"Warning: Resume path not found: {resume_path}")
            print("Starting fresh without pretrained segmentation weights")
    else:
        print("No pretrained segmentation weights provided")

    # Freeze temporal branch

    # ------------------------------
    # TensorBoard
    # ------------------------------

    if model == "stunet":
        log_dir = os.path.join(save_dir, 'STUNet_runs',
                                datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir, comment='-stunet')
    else:
        log_dir = os.path.join(save_dir, 'STUNet3Plus_runs',
                                datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir, comment='-stunet3plus')

    net.to(device)
    netD.to(device)

    # Params count
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}\n")

    # ------------------------------
    # Losses
    # ------------------------------
    lp_function = nn.MSELoss().to(device)
    criterion = nn.BCELoss().to(device)
    seg_criterion = nn.BCEWithLogitsLoss().to(device)

    # ------------------------------
    # Optimizers
    # ------------------------------
    seg_encoder_params = list(net.seg_encoder.parameters())
    seg_decoder_params = list(net.seg_decoder.parameters())

    optimizer = optim.Adam([
        {'params': seg_encoder_params, 'lr': seg_lr * 0.1},  # lower LR for encoder
        {'params': seg_decoder_params, 'lr': seg_lr},
    ], weight_decay=wd)

    optimizerG = optim.Adam([
        {'params': net.pred_encoder.parameters(), 'lr': pred_lr},
        {'params': net.pred_decoder.parameters(), 'lr': pred_lr},
    ], lr=pred_lr, weight_decay=wd)

    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, weight_decay=wd)

    print(f"Optimizer setup:")
    print(f"  Segmentation encoder LR: {seg_lr * 0.1:.6f}")
    print(f"  Segmentation decoder LR: {seg_lr:.6f}")
    print(f"  Temporal branch LR: {pred_lr:.6f} (frozen)\n")

    # ------------------------------
    # Data
    # ------------------------------
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
    ])

    if dataset_type.lower() == 'davis':
        print("Loading DAVIS dataset for pretraining...")
        db_train = davis.DAVISDataset(
            inputRes=(256, 256),
            samples_list_file=os.path.join('/home/c43n256/REU2026/SpatiotemporalFireSegmentation/stcnn/data/DAVIS16_samples_list.txt'),
            transform=composed_transforms,
            num_frame=num_frame
        )
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = None  # No testing for DAVIS

    elif dataset_type.lower() == 'fire':
        print("Loading FIRE dataset for training...")
        db_train = db.FIREDatasetRandom(
            inputRes=(256, 256),
            transform=composed_transforms,
            mode="train",
            num_frame=num_frame
        )
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4)

        print("Loading FIRE test set...")
        test_set = db.FIREDatasetRandom(
            inputRes=(256, 256),
            mode="test",
            num_frame=num_frame
        )
        test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=True)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'davis' or 'fire'")

    num_img_tr = len(trainloader)

    print(f"Training samples: {len(db_train)}")
    print(f"Training batches: {num_img_tr}")
    if test_loader:
        print(f"Test samples: {len(test_set)}")
    print(f"\nStarting training...\n")

    # ------------------------------
    # Training Loop
    # ------------------------------
    epoch_losses = []
    val_loss_list = []
    val_iou_list = []
    lp_loss = None

    for epoch in range(resume_epoch, nEpochs):
        epoch_loss = 0.0
        num_batches = len(trainloader)
        start_time = timeit.default_timer()

        net.train()

        for ii, sample_batched in enumerate(trainloader):
            seqs = sample_batched['images']      # [B, T*C, H, W]
            frames = sample_batched['frame']     # [B, C, H, W]
            gts = sample_batched['seg_gt']       # [B, 1, H, W]
            pred_gts = sample_batched['pred_gt'] # [B, C, H', W'] (temporal target)

            # Requires grad for potential GAN updates
            seqs.requires_grad_()
            frames.requires_grad_()

            seqs = seqs.to(device)
            frames = frames.to(device)
            gts = gts.to(device)
            pred_gts = pred_gts.to(device)

            # bring to fixed sizes for D / LP
            pred_gts_small = F.interpolate(pred_gts, size=(75, 75), mode='bilinear', align_corners=False)
            pred_gts_lp = F.interpolate(pred_gts, size=(100, 178), mode='bilinear', align_corners=False).detach()

            # Forward pass
            seg_res, pred = net.forward(seqs, frames)  # returns ((seg_logits, attention_outs), pred)

            # Unpack
            if isinstance(seg_res, (list, tuple)):
                seg_res = seg_res[0]  # logits

            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            D_real_input = pred_gts_small
            D_fake_input = F.interpolate(pred.detach(), size=(75, 75), mode='bilinear', align_corners=False)

            # Discriminator pass
            netD.eval()
            D_real = netD(D_real_input).squeeze(1)
            D_fake = netD(D_fake_input).squeeze(1)
            netD.train()

            # Labels shaped to current batch
            real_label = torch.ones_like(D_real)
            fake_label = torch.zeros_like(D_fake)

            # D losses
            errD_real = criterion(D_real, real_label)
            errD_fake = criterion(D_fake, fake_label)

            # ---- Update segmentation (main) ----
            optimizer.zero_grad()

            # Sanity checks
            if torch.isnan(seg_res).any() or torch.isinf(seg_res).any():
                print(f"Warning: NaN/Inf detected in seg_res at batch {ii}, skipping")
                continue

            seg_loss = seg_criterion(seg_res, gts)

            if torch.isnan(seg_loss) or torch.isinf(seg_loss):
                print(f"Warning: NaN/Inf loss at batch {ii}, skipping")
                continue

            seg_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += seg_loss.item()

            # ---- Update D (optional) ----
            if updateD:
                netD.zero_grad()
                d_loss = errD_fake + errD_real
                d_loss.backward()
                optimizerD.step()

            # ---- Update G (temporal) (optional) ----
            if updateG:
                optimizerG.zero_grad()

                netD.eval()
                D_fake = netD(D_fake_input).squeeze(1)
                netD.train()
                errG = criterion(D_fake, real_label)

                pred_for_lp = pred
                if pred_for_lp.shape[-2:] != pred_gts_lp.shape[-2:]:
                    pred_for_lp = F.interpolate(pred_for_lp, size=pred_gts_lp.shape[-2:], mode='bilinear', align_corners=False)

                lp_loss = lp_function(pred_for_lp, pred_gts_lp)
                total_loss = lp_loss + beta * errG
                total_loss.backward()
                optimizerG.step()

                if (errD_fake.data < margin).all() or (errD_real.data < margin).all():
                    updateD = False
                if (errD_fake.data > (1. - margin)).all() or (errD_real.data > (1. - margin)).all():
                    updateG = False
                if not updateD and not updateG:
                    updateD = True
                    updateG = True

            if (ii + num_img_tr * epoch) % 5 == 4 and lp_loss is not None:
                print(
                    "Iters: [%2d] time: %4.4f, lp_loss: %.8f, seg_loss: %.8f"
                    % (ii + num_img_tr * epoch, timeit.default_timer() - start_time,
                       lp_loss.item(), seg_loss.item())
                )
                print('updateD:', updateD, 'updateG:', updateG)

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch + 1}/{nEpochs}] - Avg Training Loss: {avg_epoch_loss:.8f}")

        # TB
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch + 1)

        # ------------------------------
        # Validation (FIRE only)
        # ------------------------------
        if test_loader is not None:
            val_loss = 0.0
            val_iou = 0.0
            net.eval()
            with torch.no_grad():
                for idx, sample in enumerate(test_loader):
                    seqs = sample['images'].to(device)
                    frames = sample['frame'].to(device)
                    gts = sample['seg_gt'].to(device)

                    seg_res, pred = net.forward(seqs, frames)
                    if isinstance(seg_res, (list, tuple)):
                        seg_res = seg_res[0]

                    seg_loss = seg_criterion(seg_res, gts)
                    val_loss += seg_loss.item()

                    # IoU
                    pred_probs = torch.sigmoid(seg_res)
                    pred_binary = (pred_probs > 0.5).float()
                    target_binary = (gts > 0.5).float()
                    intersection = (pred_binary * target_binary).sum()
                    union = (pred_binary + target_binary).clamp(0, 1).sum()
                    iou = (intersection / (union + 1e-6)).item()
                    val_iou += iou

            net.train()

            num_samples = len(test_loader)
            avg_val_loss = val_loss / max(1, num_samples)
            avg_val_iou = val_iou / max(1, num_samples)
            val_loss_list.append(avg_val_loss)
            val_iou_list.append(avg_val_iou)

            print(f"Epoch [{epoch + 1}/{nEpochs}] - Avg Validation Loss: {avg_val_loss:.8f}, IoU: {avg_val_iou:.4f}")

            writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
            writer.add_scalar('IoU/Validation', avg_val_iou, epoch + 1)

        # ------------------------------
        # Save snapshot
        # ------------------------------
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            save_path = os.path.join(save_model_dir, f'{modelName}-{epoch}.pth')
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': {
                    'arch': 'STUNet_UNet',
                    'num_classes': 1,
                    'num_frame': num_frame
                }
            }
            torch.save(checkpoint, save_path)
            print(f"Model saved: {save_path}")

    # ------------------------------
    # Save final
    # ------------------------------
    final_save_path = os.path.join(save_model_dir, f'{modelName}-final.pth')
    final_checkpoint = {
        'epoch': nEpochs,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': {
            'arch': 'STUNet_UNet',
            'num_classes': 1,
            'num_frame': num_frame
        },
        'train_losses': epoch_losses,
        'val_losses': val_loss_list,
        'val_ious': val_iou_list
    }
    torch.save(final_checkpoint, final_save_path)
    print(f"Final model saved: {final_save_path}")

    # ------------------------------
    # Plot curves
    # ------------------------------
    fig, axes = plt.subplots(1, 2 if test_loader else 1, figsize=(15 if test_loader else 8, 6))

    if test_loader:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # training loss
    ax1.plot(range(resume_epoch, nEpochs), epoch_losses, marker='o', linestyle='-', label="Training Loss")
    if test_loader is not None and len(val_loss_list) > 0:
        ax1.plot(range(resume_epoch, nEpochs), val_loss_list, marker='s', linestyle='--', label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss - {modelName}")
    ax1.legend()
    ax1.grid(True)

    # val IoU
    if test_loader is not None and len(val_iou_list) > 0:
        ax2.plot(range(resume_epoch, nEpochs), val_iou_list, marker='o', linestyle='-', label="Validation IoU")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("IoU")
        ax2.set_title(f"Validation IoU - {modelName}")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_model_dir, f"training_curve_{modelName}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curve saved: {plot_path}")
    plt.close()

    writer.close()
    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Model: {modelName}")
    print(f"Final training loss: {epoch_losses[-1]:.8f}")
    if test_loader and val_loss_list:
        print(f"Final validation loss: {val_loss_list[-1]:.8f}")
        print(f"Best validation IoU: {max(val_iou_list):.4f}")
    print(f"{'=' * 60}\n")


def initialize_netD(netD, model_path):
    """Initialize discriminator with pretrained Inception-v3 weights (best effort)."""
    try:
        hub_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        hub_model.eval()

        pretrained_dict = hub_model.state_dict()
        model_dict = netD.state_dict()

        # Filter out fc layers to avoid size mismatch
        filtered_dict = {k: v for k, v in pretrained_dict.items()
                         if k in model_dict and not k.startswith('fc.')}

        model_dict.update(filtered_dict)
        netD.load_state_dict(model_dict)
        print("NetD initialized with pretrained Inception-v3 weights")
    except Exception as e:
        print(f"Warning: Could not load pretrained Inception-v3: {e}")
        print("Using random initialization for NetD")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ST-UNet (UNet encoder/decoder + Attention)")

    parser.add_argument("--epochs", type=int, default=201,
                        help="Number of epochs to train (default: 201)")

    parser.add_argument("--model", type=str, default="stunet3plus", choices=["stunet", "stunet3plus"],
                        help="Architecture choice for training pipeline: 'stunet' or 'stunet3plus'")

    parser.add_argument("--frame_nums", type=int, default=4,
                        help="Number of input frames (temporal branch)")

    parser.add_argument("--dataset", type=str, default="fire", choices=["davis", "fire"],
                        help="Dataset to use: 'davis' for pretraining (no testing), 'fire' for full training (with testing)")

    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="Epoch to resume from (0 = start fresh)")

    parser.add_argument("--pretrained_seg", type=str, default=None,
                        help="Path to pretrained segmentation checkpoint")

    parser.add_argument("--output_dir", type=str, default = "/home/c43n256/REU2026/SpatiotemporalFireSegmentation/stcnn/output",
                        help="Directory to save models and logs")

    args = parser.parse_args()
    main(args)
