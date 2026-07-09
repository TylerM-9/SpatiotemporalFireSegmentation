"""
ST-DEEPLABV3+ TRAINING SCRIPT
Complete training script with dataset selection flag.
Supports DAVIS (pretraining, no testing) and FIRE (with testing).
"""

from __future__ import absolute_import, division, print_function

import argparse
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

# Import ST-DeepLabV3+ modules
from stdeeplabv3plus import create_stdeeplabv3plus, STDeepLabV3Plus
from network.joint_pred_seg import FramePredDecoder, FramePredEncoder
from network.googlenet import Inception3

from dataloaders import custom_transforms as tr
from dataloaders import DAVIS_dataloader as davis
from dataloaders import FIRE_dataloader as db
from mypath import Path

gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")


def main(args):
    # Select which GPU, -1 if CPU
    if torch.cuda.is_available():
        print(f"CUDA available, using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("CUDA not available, using CPU.")

    # Setting parameters
    resume_epoch = args.resume_epoch
    nEpochs = 201
    batch_size = 6
    snapshot = 5
    pred_lr = 1e-8  # Very low LR for frozen temporal branch
    seg_lr = 1e-4   # Main learning rate for segmentation branch
    lr_D = 1e-4
    wd = 5e-4
    beta = 0.001
    margin = 0.3

    updateD = True
    updateG = False
    num_frame = args.frame_nums
    dataset_type = args.dataset  # 'davis' or 'fire'

    # DeepLabV3+ specific parameters
    backbone = args.backbone  # 'resnet50' or 'resnet101'
    output_stride = args.output_stride  # 8 or 16
    input_size = 256

    modelName = f'STDEEPLABV3PLUS_{backbone.upper()}_{dataset_type.upper()}{num_frame}'
    resume_path_model = f'/home/c43n256/STCNN/output/STDEEPLABV3PLUS_{backbone.upper()}_{dataset_type.upper()}{num_frame}/STDEEPLABV3PLUS_{backbone.upper()}_{dataset_type.upper()}{num_frame}-199.pth'

    save_dir = Path.save_root_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_model_dir = os.path.join(save_dir, modelName)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    print(f"\n{'=' * 60}")
    print(f"Training Configuration:")
    print(f"  Dataset: {dataset_type.upper()}")
    print(f"  Model: ST-DeepLabV3+")
    print(f"  Backbone: {backbone}")
    print(f"  Output Stride: {output_stride}")
    print(f"  Input Size: {input_size}x{input_size}")
    print(f"  Frames: {num_frame}")
    print(f"  Epochs: {nEpochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Resume from epoch: {resume_epoch}")
    print(f"{'=' * 60}\n")

    # ============================================================================
    # NETWORK DEFINITION
    # ============================================================================
    netD = Inception3(num_classes=1, aux_logits=False, transform_input=True)
    initialize_netD(netD, os.path.join(
        '/home/c43n256/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4',
        'NetD_epoch-99.pth'))

    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()

    # Always load pretrained temporal weights first
    print("Loading weights from pretrained NetG")
    pretrained_netG_dict = torch.load(
        os.path.join(
            '/home/c43n256/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4',
            'NetG_epoch-99.pth'),
        map_location=torch.device(device))

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

    # Create ST-DeepLabV3+ model
    print(f"Creating ST-DeepLabV3+ with {backbone} backbone")
    net = create_stdeeplabv3plus(
        pred_enc=pred_enc,
        pred_dec=pred_dec,
        num_frame=num_frame,
        num_classes=1,
        backbone=backbone,
        output_stride=output_stride,
        input_size=input_size
    )

    # Load pretrained segmentation weights if provided
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
            print("Continuing with ImageNet pretrained backbone only")
    elif resume_epoch > 0:
        # Try to load from standard resume path
        resume_path = os.path.join(resume_path_model)
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
            print("Starting with ImageNet pretrained backbone only")
    else:
        print("No pretrained segmentation weights provided")
        print("Using ImageNet pretrained backbone only")

    # Always freeze temporal branch
    net.freeze_temporal_branch()
    print("Temporal branch frozen")

    # Freeze BatchNorm for stable training
    net.freeze_bn()
    print("BatchNorm layers frozen")

    # ============================================================================
    # SETUP TENSORBOARD
    # ============================================================================
    log_dir = os.path.join(save_dir, 'STDeepLabV3Plus_runs',
                           datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir, comment='-stdeeplabv3plus')

    net.to(device)
    netD.to(device)

    # Print parameter counts
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}\n")

    # ============================================================================
    # LOSS FUNCTIONS
    # ============================================================================
    lp_function = nn.MSELoss().to(device)
    criterion = nn.BCELoss().to(device)
    seg_criterion = nn.BCEWithLogitsLoss().to(device)

    # ============================================================================
    # OPTIMIZERS
    # ============================================================================
    # Get parameters from spatial segmentation branch only
    # Temporal branch is frozen
    seg_backbone_params = list(net.seg_model.get_backbone_params())
    seg_decoder_params = list(net.seg_model.get_decoder_params())

    optimizer = optim.SGD([
        {'params': seg_backbone_params, 'lr': seg_lr * 0.1},  # Lower LR for backbone
        {'params': seg_decoder_params, 'lr': seg_lr},
    ], weight_decay=wd, momentum=0.9)

    # Optimizer for temporal branch (if unfrozen)
    optimizerG = optim.Adam([
        {'params': net.pred_encoder.parameters(), 'lr': pred_lr},
        {'params': net.pred_decoder.parameters(), 'lr': pred_lr},
    ], lr=pred_lr, weight_decay=wd)

    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, weight_decay=wd)

    print(f"Optimizer setup:")
    print(f"  Segmentation backbone LR: {seg_lr * 0.1:.6f}")
    print(f"  Segmentation decoder LR: {seg_lr:.6f}")
    print(f"  Temporal branch LR: {pred_lr:.6f} (frozen)\n")

    # ============================================================================
    # DATA LOADERS
    # ============================================================================
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
    ])

    if dataset_type.lower() == 'davis':
        print("Loading DAVIS dataset for pretraining...")
        db_train = davis.DAVISDataset(
            inputRes=(input_size, input_size),
            samples_list_file=os.path.join('/home/c43n256/STCNN/data/DAVIS16_samples_list.txt'),
            transform=composed_transforms,
            num_frame=num_frame
        )
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = None  # No testing for DAVIS

    elif dataset_type.lower() == 'fire':
        print("Loading FIRE dataset for training...")
        db_train = db.FIREDatasetRandom(
            inputRes=(input_size, input_size),
            transform=composed_transforms,
            mode="train",
            num_frame=num_frame
        )
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4)

        print("Loading FIRE test set...")
        test_set = db.FIREDatasetRandom(
            inputRes=(input_size, input_size),
            mode="test",
            num_frame=num_frame
        )
        test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=True)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'davis' or 'fire'")

    num_img_tr = len(trainloader)
    iter_num = nEpochs * num_img_tr
    curr_iter = resume_epoch * num_img_tr

    print(f"Training samples: {len(db_train)}")
    print(f"Training batches: {num_img_tr}")
    if test_loader:
        print(f"Test samples: {len(test_set)}")
    print(f"\nStarting training...\n")

    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    epoch_losses = []
    val_loss_list = []
    val_iou_list = []
    lp_loss = None

    for epoch in range(resume_epoch, nEpochs):
        epoch_loss = 0
        num_batches = len(trainloader)
        start_time = timeit.default_timer()

        # Set to training mode but keep BN frozen
        net.train()
        net.freeze_bn()

        for ii, sample_batched in enumerate(trainloader):
            seqs = sample_batched['images']
            frames = sample_batched['frame']
            gts = sample_batched['seg_gt']
            pred_gts = sample_batched['pred_gt']

            # Forward-Backward of the mini-batch
            seqs.requires_grad_()
            frames.requires_grad_()

            seqs = seqs.to(device)
            frames = frames.to(device)
            gts = gts.to(device)
            pred_gts = pred_gts.to(device)

            pred_gts = F.interpolate(pred_gts, size=(100, 178), mode='bilinear', align_corners=False)
            pred_gts = pred_gts.detach()

            # Forward pass through ST-DeepLabV3+
            # Returns: seg_output, pred_frame, attention_features
            seg_res, pred, attention = net.forward(seqs, frames)

            D_real_input = F.interpolate(pred_gts, size=(75, 75), mode='bilinear', align_corners=False)
            D_fake_input = F.interpolate(pred.detach(), size=(75, 75), mode='bilinear', align_corners=False)

            # Compute discriminator outputs
            netD.eval()
            D_real = netD(D_real_input).squeeze(1)
            D_fake = netD(D_fake_input).squeeze(1)
            netD.train()

            # Labels that match the current batch size
            real_label = torch.ones_like(D_real)
            fake_label = torch.zeros_like(D_fake)

            # Compute discriminator losses
            errD_real = criterion(D_real, real_label)
            errD_fake = criterion(D_fake, fake_label)

            # Update segmentation network (main loss)
            optimizer.zero_grad()
            seg_loss = seg_criterion(seg_res, gts)

            # Optional: Add auxiliary loss on attention features
            if attention is not None and args.use_attention_loss:
                # Downsample ground truth to match attention feature size
                gts_down = F.interpolate(gts, size=attention.shape[-2:], mode='bilinear', align_corners=False)
                aux_loss = seg_criterion(attention, gts_down)
                total_seg_loss = seg_loss + 0.4 * aux_loss
            else:
                total_seg_loss = seg_loss

            total_seg_loss.backward()
            optimizer.step()
            curr_iter += 1

            epoch_loss += seg_loss.item()

            if updateD:
                # Update D network
                netD.zero_grad()
                d_loss = errD_fake + errD_real
                d_loss.backward()
                optimizerD.step()

            if updateG:
                optimizerG.zero_grad()

                netD.eval()
                D_fake = netD(D_fake_input).squeeze(1)
                netD.train()
                errG = criterion(D_fake, real_label)

                if pred.shape[-2:] != pred_gts.shape[-2:]:
                    pred = F.interpolate(pred, size=pred_gts.shape[-2:], mode='bilinear', align_corners=False)

                lp_loss = lp_function(pred, pred_gts)
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

            if (ii + num_img_tr * epoch) % 5 == 4 and lp_loss:
                print(
                    "Iters: [%2d] time: %4.4f, lp_loss: %.8f, G_loss: %.8f, seg_loss: %.8f"
                    % (ii + num_img_tr * epoch, timeit.default_timer() - start_time,
                       lp_loss.item(), errG.item(), seg_loss.item())
                )
                print('updateD:', updateD, 'updateG:', updateG)

        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch + 1}/{nEpochs}] - Avg Training Loss: {avg_epoch_loss:.8f}")

        # Log to tensorboard
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch + 1)

        # ============================================================================
        # VALIDATION (only for FIRE dataset)
        # ============================================================================
        if test_loader is not None:
            val_loss = 0
            val_iou = 0
            net.eval()
            with torch.no_grad():
                for idx, sample in enumerate(test_loader):
                    seqs = sample['images'].to(device)
                    frames = sample['frame'].to(device)
                    gts = sample['seg_gt'].to(device)

                    seg_res, pred, attention = net.forward(seqs, frames)

                    seg_loss = seg_criterion(seg_res, gts)
                    val_loss += seg_loss.item()

                    # Calculate IoU
                    pred_probs = torch.sigmoid(seg_res)
                    pred_binary = (pred_probs > 0.5).float()
                    target_binary = (gts > 0.5).float()
                    intersection = (pred_binary * target_binary).sum()
                    union = (pred_binary + target_binary).clamp(0, 1).sum()
                    iou = (intersection / (union + 1e-6)).item()
                    val_iou += iou

            net.train()
            net.freeze_bn()

            num_samples = len(test_loader)
            avg_val_loss = val_loss / num_samples
            avg_val_iou = val_iou / num_samples
            val_loss_list.append(avg_val_loss)
            val_iou_list.append(avg_val_iou)

            print(f"Epoch [{epoch + 1}/{nEpochs}] - Avg Validation Loss: {avg_val_loss:.8f}, IoU: {avg_val_iou:.4f}")

            # Log to tensorboard
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
            writer.add_scalar('IoU/Validation', avg_val_iou, epoch + 1)

        # ============================================================================
        # SAVE MODEL
        # ============================================================================
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            save_path = os.path.join(save_model_dir, f'{modelName}-{epoch}.pth')
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': {
                    'backbone': backbone,
                    'output_stride': output_stride,
                    'input_size': input_size,
                    'num_classes': 1,
                }
            }
            torch.save(checkpoint, save_path)
            print(f"Model saved: {save_path}")

    # ============================================================================
    # SAVE FINAL MODEL
    # ============================================================================
    final_save_path = os.path.join(save_model_dir, f'{modelName}-final.pth')
    final_checkpoint = {
        'epoch': nEpochs,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': {
            'backbone': backbone,
            'output_stride': output_stride,
            'input_size': input_size,
            'num_classes': 1,
        },
        'train_losses': epoch_losses,
        'val_losses': val_loss_list,
        'val_ious': val_iou_list
    }
    torch.save(final_checkpoint, final_save_path)
    print(f"Final model saved: {final_save_path}")

    # ============================================================================
    # PLOT TRAINING CURVES
    # ============================================================================
    fig, axes = plt.subplots(1, 2 if test_loader else 1, figsize=(15 if test_loader else 8, 6))

    if test_loader:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # Plot training loss
    ax1.plot(range(resume_epoch, nEpochs), epoch_losses, marker='o', linestyle='-', label="Training Loss")
    if test_loader is not None and len(val_loss_list) > 0:
        ax1.plot(range(resume_epoch, nEpochs), val_loss_list, marker='s', linestyle='--',
                 label="Validation Loss", color='r')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss - {modelName}")
    ax1.legend()
    ax1.grid(True)

    # Plot validation IoU if available
    if test_loader is not None and len(val_iou_list) > 0:
        ax2.plot(range(resume_epoch, nEpochs), val_iou_list, marker='o', linestyle='-',
                 label="Validation IoU", color='g')
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
    """Initialize discriminator with pretrained Inception-v3 weights."""
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
    parser = argparse.ArgumentParser(description="Train ST-DeepLabV3+ with attention")

    parser.add_argument("--frame_nums", type=int, default=4,
                        help="Number of input frames")

    parser.add_argument("--dataset", type=str, default="fire", choices=["davis", "fire"],
                        help="Dataset to use: 'davis' for pretraining (no testing), 'fire' for full training (with testing)")

    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="Epoch to resume from (0 = start fresh)")

    parser.add_argument("--pretrained_seg", type=str, default=None,
                        help="Path to pretrained segmentation checkpoint (loads both temporal and spatial)")

    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"],
                        help="Backbone architecture for DeepLabV3+")

    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16],
                        help="Output stride for DeepLabV3+ (8 or 16)")

    parser.add_argument("--use_attention_loss", action="store_true",
                        help="Use auxiliary loss on attention features")

    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with gradient checking")

    args = parser.parse_args()

    # Enable anomaly detection if debug mode
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        print("Debug mode enabled: Anomaly detection ON")

    main(args)