"""
Testing script for SegBranch (SegEncoder + SegDecoder) checkpoints
Matches your SegBranch multi-round training file.

You ONLY set:
  - model_path (path to *_best.pth / *_final.pth / *_epoch*.pth)
Then run evaluation on FIRE test split with threshold sweep, save examples + grid, and write results txt.

Assumptions (match your training dataloaders):
  - FIRE sample keys: 'frame', 'seg_gt'
  - Input frame is normalized like your training visualization assumed
  - Net forward returns a LIST of multi-scale logits; use pred[-1] for evaluation
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from mypath import Path
from dataloaders import FIRE_dataloader as db

from network.joint_pred_seg import SegBranch, SegDecoder, SegEncoder
from torchvision.models import resnet101, ResNet101_Weights


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# UPDATE THIS PATH
# =========================
model_path = "/home/c43n256/STCNN/output/SegBranch_FIRE_round2/SegBranch_FIRE_round2_best.pth"
model_name = "SegBranch_Test"


# -------------------------
# Model loading (same init as training)
# -------------------------
def load_segbranch_model(model_path, device):
    print(f"Loading SegBranch model from: {model_path}")

    encoder = SegEncoder()

    # IMPORTANT: training initializes with ImageNet ResNet101 weights BEFORE optionally loading checkpoint
    print("Initializing encoder with ResNet101 ImageNet weights...")
    pretrained_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained_model.state_dict()
    model_dict = encoder.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    encoder.load_state_dict(model_dict)

    decoder = SegDecoder()
    net = SegBranch(net_enc=encoder, net_dec=decoder)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)

    print("\n" + "=" * 60)
    print("CHECKPOINT ANALYSIS")
    print("=" * 60)

    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("Loading from 'state_dict' key")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
    else:
        state_dict = checkpoint
        print("Checkpoint is a state_dict")

    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"WARNING - Missing keys ({len(missing_keys)}): {missing_keys[:8]} ...")
    if unexpected_keys:
        print(f"WARNING - Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:8]} ...")
    if not missing_keys and not unexpected_keys:
        print("✓ All weights loaded successfully!")

    print("=" * 60 + "\n")

    net.to(device)
    net.eval()
    return net


# -------------------------
# Metrics (same as your other testers)
# -------------------------
def precision_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    tp = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
    fp = np.logical_and(y_true_bin == 0, y_pred_bin == 1).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    tp = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
    fn = np.logical_and(y_true_bin == 1, y_pred_bin == 0).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def iou_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    inter = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()
    return inter / union if union != 0 else 1.0


def iou_score_class(y_true, y_pred, threshold=0.5, target_class=1):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)

    if target_class == 0:
        y_true_c = (y_true_bin == 0).astype(np.uint8)
        y_pred_c = (y_pred_bin == 0).astype(np.uint8)
    else:
        y_true_c = y_true_bin
        y_pred_c = y_pred_bin

    inter = np.logical_and(y_true_c, y_pred_c).sum()
    union = np.logical_or(y_true_c, y_pred_c).sum()
    return inter / union if union != 0 else 1.0


def pixel_accuracy(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    correct = (y_true_bin == y_pred_bin).sum()
    total = y_true_bin.size
    return correct / total if total > 0 else 1.0


def dice_loss(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    inter = (y_true_bin * y_pred_bin).sum()
    dice = (2.0 * inter + smooth) / (y_true_bin.sum() + y_pred_bin.sum() + smooth)
    return 1 - dice


# -------------------------
# Saving examples (input + GT + pred overlay)
# -------------------------
def save_example_image_seg(input_img, gt_sample, seg_pred, index,
                           iou, dice, precision, recall, save_dir, threshold):
    # input_img: torch tensor [B,C,H,W] (from 'frame')
    if torch.is_tensor(input_img):
        input_numpy = input_img.detach().cpu().numpy()
    else:
        input_numpy = input_img

    if len(input_numpy.shape) == 4:
        input_frame = input_numpy[0].transpose(1, 2, 0)
    elif len(input_numpy.shape) == 3:
        input_frame = input_numpy.transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected input shape: {input_numpy.shape}")

    # Try to make it displayable (your training vis did min-max normalization)
    input_frame = input_frame.astype(np.float32)
    denom = max((input_frame.max() - input_frame.min()), 1e-8)
    input_vis = (input_frame - input_frame.min()) / denom
    input_vis = np.clip(input_vis, 0, 1)

    gt_display = np.squeeze(gt_sample)
    pred_binary = (seg_pred > threshold).astype(np.float32)
    pred_display = np.squeeze(pred_binary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_vis)
    axes[0].set_title("Input", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(input_vis)
    axes[1].imshow(gt_display, alpha=0.5, cmap="jet")
    axes[1].set_title("GT", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(input_vis)
    axes[2].imshow(pred_display, alpha=0.5, cmap="jet")
    axes[2].set_title(
        f"Pred\nIoU:{iou:.4f} Dice:{1-dice:.4f}\nP:{precision:.4f} R:{recall:.4f}",
        fontsize=11, fontweight="bold"
    )
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"example_{index:04d}_iou{iou:.3f}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_comparison_grid(examples_dir, num_examples=9):
    example_files = sorted(glob.glob(os.path.join(examples_dir, "example_*.png")))
    if len(example_files) == 0:
        print(f"No example images found in: {examples_dir}")
        return

    example_files = example_files[:num_examples]
    n_cols = 3
    n_rows = (len(example_files) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(example_files):
        row = idx // n_cols
        col = idx % n_cols
        img = mpimg.imread(img_path)
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

    for idx in range(len(example_files), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(examples_dir, "examples_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved example grid to: {grid_path}")


# -------------------------
# Main testing
# -------------------------
def main(epochs_tag=0, test_thresholds=None, out_dir_name=None):
    if test_thresholds is None:
        test_thresholds = [0.5]

    save_root = Path.save_root_dir()

    if out_dir_name is None:
        out_dir_name = model_name

    save_model_dir = os.path.join(save_root, out_dir_name)
    os.makedirs(save_model_dir, exist_ok=True)

    model = load_segbranch_model(model_path, device)
    if model is None:
        return

    # FIRE test set (num_frame=1 like training)
    test_set = db.FIREDataset(
        inputRes=(256, 256),
        samples_path="/home/c43n256/Data/Mask_Data",
        transform=None,
        mode="test",
        num_frame=1
    )

    testloader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)
    print(f"Total dataset size: {len(testloader)} images")

    for threshold in test_thresholds:
        print(f"\n{'=' * 60}")
        print(f"Testing with threshold: {threshold}")
        print(f"{'=' * 60}")

        examples_dir = os.path.join(save_model_dir, f"examples_epoch{epochs_tag}_t{threshold:.1f}")
        os.makedirs(examples_dir, exist_ok=True)

        total_fg_iou = 0.0
        total_pa = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_recall = 0.0
        per_class_iou = []
        processed_count = 0

        num_examples_to_save = 20
        save_interval = max(1, len(testloader) // num_examples_to_save)

        with torch.no_grad():
            for ii, sample_batched in enumerate(testloader):
                frames = sample_batched["frame"].to(device)
                gts = sample_batched["seg_gt"]

                # forward: list of logits (multi-scale)
                pred_list = model.forward(frames)
                logits = pred_list[-1] if isinstance(pred_list, list) else pred_list

                seg_pred = torch.sigmoid(logits)
                seg_pred = seg_pred[0].cpu().numpy()  # (C,H,W)

                gt_sample = gts[0].cpu().numpy().transpose([1, 2, 0])
                if gt_sample.max() > 1.0:
                    gt_sample = gt_sample / 255.0

                cur_iou = iou_score(gt_sample, seg_pred, threshold)
                cur_pa = pixel_accuracy(gt_sample, seg_pred, threshold)
                cur_dice = dice_loss(gt_sample, seg_pred, threshold)
                cur_prec = precision_score(gt_sample, seg_pred, threshold)
                cur_rec = recall_score(gt_sample, seg_pred, threshold)

                bg_iou = iou_score_class(gt_sample, seg_pred, threshold, target_class=0)
                fg_iou = iou_score_class(gt_sample, seg_pred, threshold, target_class=1)
                per_class_iou.append([bg_iou, fg_iou])

                total_fg_iou += cur_iou
                total_pa += cur_pa
                total_dice += cur_dice
                total_precision += cur_prec
                total_recall += cur_rec
                processed_count += 1

                if ii % save_interval == 0 and ii < num_examples_to_save * save_interval:
                    save_example_image_seg(
                        frames, gt_sample, seg_pred, ii,
                        cur_iou, cur_dice, cur_prec, cur_rec,
                        examples_dir, threshold
                    )

        if processed_count == 0:
            print("No images processed!")
            continue

        mean_fg_iou = total_fg_iou / processed_count
        mean_pa = total_pa / processed_count
        mean_dice_loss = total_dice / processed_count
        mean_precision = total_precision / processed_count
        mean_recall = total_recall / processed_count

        per_class_iou = np.array(per_class_iou)
        mean_bg_iou = float(np.mean(per_class_iou[:, 0]))
        mean_fg_iou_class = float(np.mean(per_class_iou[:, 1]))
        mean_iou_classes = (mean_bg_iou + mean_fg_iou_class) / 2.0

        f1 = (2 * mean_precision * mean_recall / (mean_precision + mean_recall)) if (mean_precision + mean_recall) > 0 else 0.0
        dice_score = 1.0 - mean_dice_loss

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Checkpoint: {model_path}")
        print(f"Dataset: {processed_count} images")
        print(f"Threshold: {threshold}")
        print("-" * 40)
        print(f"IoU (Foreground):       {mean_fg_iou:.4f}")
        print(f"Mean IoU (All Classes): {mean_iou_classes:.4f}")
        print(f"  - Background IoU:     {mean_bg_iou:.4f}")
        print(f"  - Foreground IoU:     {mean_fg_iou_class:.4f}")
        print(f"Mean Pixel Accuracy:    {mean_pa:.4f}")
        print(f"Precision:              {mean_precision:.4f}")
        print(f"Recall:                 {mean_recall:.4f}")
        print(f"F1 Score:               {f1:.4f}")
        print(f"Dice Score:             {dice_score:.4f}")
        print(f"Dice Loss:              {mean_dice_loss:.4f}")
        print("=" * 60)

        results_file = os.path.join(save_model_dir, f"evaluation_results_t{threshold:.1f}.txt")
        with open(results_file, "w") as f:
            f.write("SegBranch Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Checkpoint: {model_path}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Dataset Size: {processed_count}\n")
            f.write("-" * 50 + "\n")
            f.write(f"IoU (Foreground):       {mean_fg_iou:.6f}\n")
            f.write(f"Mean IoU (All Classes): {mean_iou_classes:.6f}\n")
            f.write(f"  - Background IoU:     {mean_bg_iou:.6f}\n")
            f.write(f"  - Foreground IoU:     {mean_fg_iou_class:.6f}\n")
            f.write(f"Mean Pixel Accuracy:    {mean_pa:.6f}\n")
            f.write(f"Precision:              {mean_precision:.6f}\n")
            f.write(f"Recall:                 {mean_recall:.6f}\n")
            f.write(f"F1 Score:               {f1:.6f}\n")
            f.write(f"Dice Score:             {dice_score:.6f}\n")
            f.write(f"Dice Loss:              {mean_dice_loss:.6f}\n")

        print(f"Results saved to: {results_file}")
        save_comparison_grid(examples_dir, num_examples=9)


if __name__ == "__main__":
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    main(epochs_tag=0, test_thresholds=test_thresholds)
