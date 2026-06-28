"""
TRAINING SCRIPT MODIFICATIONS
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

from network.ResUNet_new import create_stcnn_with_attention
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
    resume_epoch = 1
    nEpochs = 201
    batch_size = 6
    snapshot = 5
    pred_lr = 1e-8
    seg_lr = 1e-4
    lr_D = 1e-4
    wd = 5e-4
    beta = 0.001
    margin = 0.3

    updateD = True
    updateG = False
    num_frame = args.frame_nums
    dataset_type = args.dataset  # 'davis' or 'fire'

    modelName = f'STCNN_frame_RESUNET{dataset_type.upper()}{num_frame}'
    resume_path_model = '/home/r56x196/STCNN/output/STCNN_frame_RESUNETDAVIS4/STCNN_frame_RESUNETDAVIS4-199.pth'

    save_dir = Path.save_root_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_model_dir = os.path.join(save_dir, modelName)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    print(f"\n{'=' * 60}")
    print(f"Training Configuration:")
    print(f"  Dataset: {dataset_type.upper()}")
    print(f"  Model: {modelName}")
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
        '/home/r56x196/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4',
        'NetD_epoch-99.pth'))

    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()

    if resume_epoch == 0:
        # Initialize temporal branch with pretrained weights
        print("Loading weights from pretrained NetG")
        pretrained_netG_dict = torch.load(
            os.path.join(
                '/home/r56x196/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4',
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

        # Create new STCNN with attention-based decoder
        net = create_stcnn_with_attention(
            pred_enc=pred_enc,
            pred_dec=pred_dec,
            num_frame=num_frame,
            encoder_name="resnet34",
            encoder_weights="imagenet",
            decoder_channels=(256, 128, 64),
            n_classes=1
        )

        # Freeze temporal branch (standard practice)
        net.freeze_temporal_branch()
    else:
        # For resuming training
        net = create_stcnn_with_attention(
            pred_enc=pred_enc,
            pred_dec=pred_dec,
            num_frame=num_frame,
            encoder_name="resnet34",
            encoder_weights="imagenet",
            decoder_channels=(256, 128, 64),
            n_classes=1
        )

        resume_path = os.path.join(resume_path_model)
        print(f"Resuming from: {resume_path}")
        net.load_state_dict(torch.load(resume_path, map_location=lambda storage, loc: storage))

    # ============================================================================
    # SETUP TENSORBOARD
    # ============================================================================
    log_dir = os.path.join(save_dir, 'JointPredSegNet_runs',
                           datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir, comment='-parent')

    net.to(device)
    netD.to(device)

    # ============================================================================
    # LOSS FUNCTIONS
    # ============================================================================
    lp_function = nn.MSELoss().to(device)
    criterion = nn.BCELoss().to(device)
    seg_criterion = nn.BCEWithLogitsLoss().to(device)

    # ============================================================================
    # OPTIMIZERS
    # ============================================================================
    optimizer = optim.SGD([
        {'params': [param for name, param in net.seg_encoder.named_parameters()], 'lr': seg_lr},
        {'params': [param for name, param in net.seg_decoder.named_parameters()], 'lr': seg_lr},
    ], weight_decay=wd, momentum=0.9)

    optimizerG = optim.Adam([
        {'params': [param for name, param in net.pred_encoder.named_parameters()], 'lr': pred_lr},
        {'params': [param for name, param in net.pred_decoder.named_parameters()], 'lr': pred_lr},
    ], lr=pred_lr, weight_decay=wd)

    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, weight_decay=wd)

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
            inputRes=(256, 256),
            samples_list_file=os.path.join('/home/r56x196/STCNN/data/DAVIS16_samples_list.txt'),
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
    lp_loss = None

    for epoch in range(resume_epoch, nEpochs):
        epoch_loss = 0
        num_batches = len(trainloader)
        start_time = timeit.default_timer()

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

            pred_gts = F.upsample(pred_gts, size=(100, 178), mode='bilinear', align_corners=False)
            pred_gts = pred_gts.detach()

            seg_res, pred = net.forward(seqs, frames)

            # Handle tuple/list return from forward()
            if isinstance(seg_res, (list, tuple)):
                seg_res = seg_res[0]
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

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

            # Update segmentation network
            optimizer.zero_grad()
            seg_loss = seg_criterion(seg_res, gts)
            seg_loss.backward()
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

        # ============================================================================
        # VALIDATION (only for FIRE dataset)
        # ============================================================================
        if test_loader is not None:
            val_loss = 0
            net.eval()
            with torch.no_grad():
                for idx, sample in enumerate(test_loader):
                    seqs = sample['images'].to(device)
                    frames = sample['frame'].to(device)
                    gts = sample['seg_gt'].to(device)

                    seg_res, pred = net.forward(seqs, frames)

                    # Handle tuple/list return
                    if isinstance(seg_res, (list, tuple)):
                        seg_res = seg_res[0]

                    seg_loss = seg_criterion(seg_res, gts)
                    val_loss += seg_loss.item()

            net.train()
            num_samples = len(test_loader)
            avg_val_loss = val_loss / num_samples
            val_loss_list.append(avg_val_loss)
            print(f"Epoch [{epoch + 1}/{nEpochs}] - Avg Validation Loss: {avg_val_loss:.8f}")

        # ============================================================================
        # SAVE MODEL
        # ============================================================================
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            save_path = os.path.join(save_model_dir, f'{modelName}-{epoch}.pth')
            torch.save(net.state_dict(), save_path)
            print(f"Model saved: {save_path}")

    # ============================================================================
    # PLOT TRAINING CURVES
    # ============================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(range(resume_epoch, nEpochs), epoch_losses, marker='o', linestyle='-', label="Training Loss")

    if test_loader is not None and len(val_loss_list) > 0:
        plt.plot(range(resume_epoch, nEpochs), val_loss_list, marker='s', linestyle='--',
                 label="Validation Loss", color='r')

    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title(f"Training {modelName}")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(save_model_dir, f"training_curve_{modelName}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curve saved: {plot_path}")

    writer.close()


def initialize_netD(netD, model_path):
    """Initialize discriminator with pretrained Inception-v3 weights."""
    hub_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    hub_model.eval()

    pretrained_dict = hub_model.state_dict()
    model_dict = netD.state_dict()

    # Filter out fc layers to avoid size mismatch
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and not k.startswith('fc.')}

    model_dict.update(filtered_dict)
    netD.load_state_dict(model_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STCNN with attention")

    parser.add_argument("--frame_nums", type=int, default=4,
                        help="Number of input frames")

    parser.add_argument("--dataset", type=str, default="fire", choices=["davis", "fire"],
                        help="Dataset to use: 'davis' for pretraining (no testing), 'fire' for full training (with testing)")

    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="Epoch to resume from (0 = start fresh)")

    args = parser.parse_args()
    main(args)