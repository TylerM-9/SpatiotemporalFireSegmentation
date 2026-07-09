"""
Testing script for STCNN Joint Prediction + Segmentation using JointSegDecoderCBAM

- Loads ONE checkpoint (best/final/epoch*.pth) and evaluates on FIRE test split
- Sweeps thresholds (like your first tester)
- Saves example images + a grid
- Writes evaluation_results_tX.txt

UPDATE:
  model_path = ".../STCNN_CBAM_FIRE4_round0/....pth"
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from mypath import Path
from dataloaders import FIRE_dataloader as db

from network.joint_pred_seg import (
    STCNN,
    FramePredEncoder,
    FramePredDecoder,
    SegEncoder,
    JointSegDecoderCBAM,  # <<< CBAM decoder
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# UPDATE THIS
# =========================
model_path = "/home/c43n256/STCNN/output/STCNN_CBAM_FIRE4_round2/STCNN_CBAM_FIRE4_round2_best.pth"
model_name = "STCNN_CBAM_JointPredSeg"


# -------------------------
# Load model
# -------------------------
def load_stcnn_cbam_model(model_path, num_frame, device):
    print(f"Loading STCNN+CBAM model from: {model_path}")

    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()
    seg_enc = SegEncoder()
    seg_dec = JointSegDecoderCBAM()

    net = STCNN(pred_enc, pred_dec, seg_enc, seg_dec)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)

    print("\n" + "=" * 60)
    print("CHECKPOINT ANALYSIS")
    print("=" * 60)

    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        print("Using dict checkpoint" + (" (state_dict)" if "state_dict" in checkpoint else " (direct)"))
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
# Metrics (same as your first tester)
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
# Visualization helpers
# -------------------------
def save_example_image_with_pred(frames, pred_frame, gt_sample, seg_pred, index,
                                 iou, dice, precision, recall, save_dir, threshold):
    """
    frames: [B,3,H,W] current frame
    pred_frame: [B,3,h,w] predicted next frame (often tanh)
    gt_sample: [H,W,1] or [H,W]
    seg_pred: [C,H,W] sigmoid probs
    """
    # input frame visualization (robust min-max)
    inp = frames.detach().cpu().numpy()
    input_frame = inp[0].transpose(1, 2, 0).astype(np.float32)
    denom = max((input_frame.max() - input_frame.min()), 1e-8)
    input_vis = (input_frame - input_frame.min()) / denom
    input_vis = np.clip(input_vis, 0, 1)

    # predicted frame visualization (assume tanh)
    pred_np = pred_frame.detach().cpu().numpy()
    pred_disp = pred_np[0].transpose(1, 2, 0).astype(np.float32)
    pred_disp = (pred_disp + 1.0) / 2.0
    pred_disp = np.clip(pred_disp, 0, 1)

    gt_display = np.squeeze(gt_sample)
    pred_binary = (seg_pred > threshold).astype(np.float32)
    pred_seg_display = np.squeeze(pred_binary)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(input_vis)
    axes[0, 0].set_title("Input Frame", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_disp)
    axes[0, 1].set_title("Predicted Next Frame", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(input_vis)
    axes[1, 0].imshow(gt_display, alpha=0.5, cmap="jet")
    axes[1, 0].set_title("Ground Truth Segmentation", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(input_vis)
    axes[1, 1].imshow(pred_seg_display, alpha=0.5, cmap="jet")
    axes[1, 1].set_title(
        f"Predicted Segmentation\nIoU:{iou:.4f} Dice:{1-dice:.4f}\nP:{precision:.4f} R:{recall:.4f}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 1].axis("off")

    fig.suptitle(f"STCNN-CBAM Sample {index} - Threshold: {threshold}", fontsize=14, fontweight="bold")
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
# Main eval
# -------------------------
def main(frame=4, epochs_tag=0, test_thresholds=None, model_round=0):
    if test_thresholds is None:
        test_thresholds = [0.5]

    save_root = Path.save_root_dir()

    model_dir_name = f"{model_name}_round{model_round}"
    save_model_dir = os.path.join(save_root, model_dir_name)
    os.makedirs(save_model_dir, exist_ok=True)

    model = load_stcnn_cbam_model(model_path, frame, device)
    if model is None:
        return

    print(f"Testing STCNN-CBAM round={model_round}")
    print(f"Thresholds: {test_thresholds}")

    test_set = db.FIREDataset(
        inputRes=(256, 256),
        mode="test",
        num_frame=frame
    )
    testloader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)
    print(f"Total dataset size: {len(testloader)} images")

    for threshold in test_thresholds:
        print(f"\n{'=' * 60}")
        print(f"Testing with threshold: {threshold}")
        print(f"{'=' * 60}")

        examples_dir = os.path.join(save_model_dir, f"examples_epoch{epochs_tag}_t{threshold:.1f}")
        os.makedirs(examples_dir, exist_ok=True)
        print(f"Saving example images to: {examples_dir}")

        total_iou = 0.0
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
                seqs = sample_batched["images"].to(device)
                frames = sample_batched["frame"].to(device)
                gts = sample_batched["seg_gt"]

                seg_res, pred = model.forward(seqs, frames)

                seg_logits = seg_res[-1] if isinstance(seg_res, list) else seg_res
                seg_prob = torch.sigmoid(seg_logits)[0].cpu().numpy()  # (C,H,W)

                gt_sample = gts[0].cpu().numpy().transpose([1, 2, 0])
                if gt_sample.max() > 1.0:
                    gt_sample = gt_sample / 255.0

                cur_iou = iou_score(gt_sample, seg_prob, threshold)
                cur_pa = pixel_accuracy(gt_sample, seg_prob, threshold)
                cur_dice = dice_loss(gt_sample, seg_prob, threshold)
                cur_prec = precision_score(gt_sample, seg_prob, threshold)
                cur_rec = recall_score(gt_sample, seg_prob, threshold)

                bg_iou = iou_score_class(gt_sample, seg_prob, threshold, target_class=0)
                fg_iou = iou_score_class(gt_sample, seg_prob, threshold, target_class=1)
                per_class_iou.append([bg_iou, fg_iou])

                total_iou += cur_iou
                total_pa += cur_pa
                total_dice += cur_dice
                total_precision += cur_prec
                total_recall += cur_rec
                processed_count += 1

                if ii % save_interval == 0 and ii < num_examples_to_save * save_interval:
                    save_example_image_with_pred(
                        frames, pred, gt_sample, seg_prob, ii,
                        cur_iou, cur_dice, cur_prec, cur_rec,
                        examples_dir, threshold
                    )

                if (ii + 1) % 100 == 0:
                    print(f"Processed {ii + 1}/{len(testloader)} images... "
                          f"IoU(avg): {total_iou / processed_count:.4f}")

        if processed_count == 0:
            print("No images processed!")
            continue

        mean_iou = total_iou / processed_count
        mean_pa = total_pa / processed_count
        mean_dice_loss = total_dice / processed_count
        mean_precision = total_precision / processed_count
        mean_recall = total_recall / processed_count

        per_class_iou = np.array(per_class_iou)
        mean_bg_iou = float(np.mean(per_class_iou[:, 0]))
        mean_fg_iou = float(np.mean(per_class_iou[:, 1]))
        mean_iou_classes = (mean_bg_iou + mean_fg_iou) / 2.0

        f1 = (2 * mean_precision * mean_recall / (mean_precision + mean_recall)) if (mean_precision + mean_recall) > 0 else 0.0
        dice_score = 1.0 - mean_dice_loss

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Checkpoint: {model_path}")
        print("Architecture: STCNN + JointSegDecoderCBAM")
        print(f"Dataset: {processed_count} images")
        print(f"Threshold: {threshold}")
        print("-" * 40)
        print(f"IoU (Foreground):       {mean_iou:.4f}")
        print(f"Mean IoU (All Classes): {mean_iou_classes:.4f}")
        print(f"  - Background IoU:     {mean_bg_iou:.4f}")
        print(f"  - Foreground IoU:     {mean_fg_iou:.4f}")
        print(f"Mean Pixel Accuracy:    {mean_pa:.4f}")
        print(f"Precision:              {mean_precision:.4f}")
        print(f"Recall:                 {mean_recall:.4f}")
        print(f"F1 Score:               {f1:.4f}")
        print(f"Dice Score:             {dice_score:.4f}")
        print(f"Dice Loss:              {mean_dice_loss:.4f}")
        print("=" * 60)

        results_file = os.path.join(save_model_dir, f"evaluation_results_t{threshold:.1f}.txt")
        with open(results_file, "w") as f:
            f.write("STCNN-CBAM Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Checkpoint: {model_path}\n")
            f.write("Architecture: STCNN + JointSegDecoderCBAM\n")
            f.write(f"Frames: {frame}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Dataset Size: {processed_count}\n")
            f.write("-" * 50 + "\n")
            f.write(f"IoU (Foreground):       {mean_iou:.6f}\n")
            f.write(f"Mean IoU (All Classes): {mean_iou_classes:.6f}\n")
            f.write(f"  - Background IoU:     {mean_bg_iou:.6f}\n")
            f.write(f"  - Foreground IoU:     {mean_fg_iou:.6f}\n")
            f.write(f"Mean Pixel Accuracy:    {mean_pa:.6f}\n")
            f.write(f"Precision:              {mean_precision:.6f}\n")
            f.write(f"Recall:                 {mean_recall:.6f}\n")
            f.write(f"F1 Score:               {f1:.6f}\n")
            f.write(f"Dice Score:             {dice_score:.6f}\n")
            f.write(f"Dice Loss:              {mean_dice_loss:.6f}\n")

        print(f"Results saved to: {results_file}")
        save_comparison_grid(examples_dir, num_examples=9)


def test_all_rounds(frame=4, epochs_tag=0, test_thresholds=None, num_rounds=3):
    """
    Convenience helper if your checkpoints follow:
      STCNN_CBAM_FIRE{frame}_round{r}/STCNN_CBAM_FIRE{frame}_round{r}_best.pth
    Adjust if your naming differs.
    """
    if test_thresholds is None:
        test_thresholds = [0.5]

    save_root = Path.save_root_dir()

    global model_path
    for round_num in range(num_rounds):
        print(f"\n{'=' * 80}\nTESTING ROUND {round_num}\n{'=' * 80}")

        model_path = os.path.join(
            save_root,
            f"STCNN_CBAM_FIRE{frame}_round{round_num}",
            f"STCNN_CBAM_FIRE{frame}_round{round_num}_best.pth"
        )

        if os.path.exists(model_path):
            main(frame=frame, epochs_tag=epochs_tag, test_thresholds=test_thresholds, model_round=round_num)
        else:
            print(f"Model not found: {model_path}")
            print("Skipping this round...")


if __name__ == "__main__":
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Option 1: test a single checkpoint (set model_path at top)
    main(frame=4, epochs_tag=0, test_thresholds=test_thresholds, model_round=0)

    # Option 2: test all rounds (uncomment)
    # test_all_rounds(frame=4, epochs_tag=0, test_thresholds=test_thresholds, num_rounds=3)
