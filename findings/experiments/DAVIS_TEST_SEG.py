"""
Testing script for STCNN Joint Prediction + Segmentation model
(Compatible with your MULTI-ROUND / DAVIS+FIRE training pipeline checkpoints)

Usage:
1) Set `model_path` to your checkpoint (.pth)
2) Run. It will:
   - load STCNN (FramePredEncoder/Decoder + SegEncoder + JointSegDecoder)
   - run evaluation on FIRE test split
   - sweep thresholds (default 0.1..0.9)
   - save example images + grid
   - save results to txt

Notes:
- Your training code saves `torch.save(net.state_dict(), path)` so this loader supports raw state_dict.
- Also supports older format: {"state_dict": ...}
"""
import glob
import numpy as np
import os
import torch
from mypath import Path
from dataloaders import FIRE_dataloader as db
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# STCNN architecture components (same as training)
from network.joint_pred_seg import (
    STCNN,
    FramePredEncoder,
    FramePredDecoder,
    SegEncoder,
    JointSegDecoder
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# UPDATE THIS
# =========================
model_path = "/home/r56x196/STCNN/output/STCNN_FIRE4_round2/STCNN_FIRE4_round2_final.pth"
model_name = "STCNN_JointPredSeg"   # just for printing + folder naming


# -------------------------
# Model loading
# -------------------------
def load_stcnn_model(model_path, num_frame, device):
    print(f"Loading STCNN model from: {model_path}")

    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()
    seg_enc = SegEncoder()
    seg_dec = JointSegDecoder()

    net = STCNN(pred_enc, pred_dec, seg_enc, seg_dec)

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
            if "epoch" in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
        else:
            # your training saves raw state_dict, but sometimes it looks like a dict anyway
            # so we treat it as state_dict
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
# Metrics
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
# Visualization saving
# -------------------------
def save_example_image_with_pred(input_img, pred_frame, gt_sample, seg_pred, index,
                                 iou, dice, precision, recall, save_dir, threshold):
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

    # assumes [-1,1]
    input_frame = (input_frame + 1.0) / 2.0
    input_frame = np.clip(input_frame, 0, 1)

    if torch.is_tensor(pred_frame):
        pred_numpy = pred_frame.detach().cpu().numpy()
    else:
        pred_numpy = pred_frame

    if len(pred_numpy.shape) == 4:
        pred_display = pred_numpy[0].transpose(1, 2, 0)
    elif len(pred_numpy.shape) == 3:
        pred_display = pred_numpy.transpose(1, 2, 0)
    else:
        pred_display = pred_numpy

    pred_display = (pred_display + 1.0) / 2.0
    pred_display = np.clip(pred_display, 0, 1)

    gt_display = np.squeeze(gt_sample)
    pred_binary = (seg_pred > threshold).astype(np.float32)
    pred_seg_display = np.squeeze(pred_binary)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(input_frame)
    axes[0, 0].set_title('Input Frame', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_display)
    axes[0, 1].set_title('Predicted Next Frame', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(input_frame)
    axes[1, 0].imshow(gt_display, alpha=0.5, cmap='jet')
    axes[1, 0].set_title('Ground Truth Segmentation', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(input_frame)
    axes[1, 1].imshow(pred_seg_display, alpha=0.5, cmap='jet')
    axes[1, 1].set_title(
        f'Predicted Segmentation\nIoU: {iou:.4f} | Dice: {1 - dice:.4f}\nP: {precision:.4f} | R: {recall:.4f}',
        fontsize=11, fontweight='bold'
    )
    axes[1, 1].axis('off')

    fig.suptitle(f'STCNN Sample {index} - Threshold: {threshold}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'example_{index:04d}_iou{iou:.3f}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_grid(examples_dir, num_examples=9):
    import matplotlib.image as mpimg

    example_files = sorted(glob.glob(os.path.join(examples_dir, 'example_*.png')))
    if len(example_files) == 0:
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
        axes[row, col].axis('off')

    for idx in range(len(example_files), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    grid_path = os.path.join(examples_dir, 'examples_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved example grid to: {grid_path}")


# -------------------------
# Main testing
# -------------------------
def main(frame=4, epochs_tag=0, test_thresholds=None, out_dir_name=None):
    if test_thresholds is None:
        test_thresholds = [0.5]

    num_frame = frame
    save_root = Path.save_root_dir()

    if out_dir_name is None:
        # same style as your first tester: folder based on model_name
        out_dir_name = model_name

    save_model_dir = os.path.join(save_root, out_dir_name)
    os.makedirs(save_model_dir, exist_ok=True)

    model = load_stcnn_model(model_path, num_frame, device)
    if model is None:
        return

    print(f"Testing thresholds: {test_thresholds}")

    for threshold in test_thresholds:
        print(f"\n{'=' * 60}")
        print(f"Testing with threshold: {threshold}")
        print(f"{'=' * 60}")

        test_set = db.FIREDataset(inputRes=(256, 256), mode="test", num_frame=num_frame)
        testloader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)

        print(f"Total dataset size: {len(testloader)} images")

        examples_dir = os.path.join(save_model_dir, f"examples_epoch{epochs_tag}_t{threshold:.1f}")
        os.makedirs(examples_dir, exist_ok=True)
        print(f"Saving example images to: {examples_dir}")

        total_iou = 0
        total_pa = 0
        total_dice = 0
        total_precision = 0
        total_recall = 0
        per_class_iou = []
        processed_count = 0

        num_examples_to_save = 20
        save_interval = max(1, len(testloader) // num_examples_to_save)

        with torch.no_grad():
            for ii, sample_batched in enumerate(testloader):
                seqs = sample_batched['images'].to(device)
                frames = sample_batched['frame'].to(device)
                gts = sample_batched['seg_gt']

                seg_res, pred = model.forward(seqs, frames)

                seg_logits = seg_res[-1] if isinstance(seg_res, list) else seg_res

                seg_pred = torch.sigmoid(seg_logits)
                seg_pred = seg_pred[0].cpu().numpy()  # (C,H,W)

                gt_sample = gts[0].cpu().numpy().transpose([1, 2, 0])
                if gt_sample.max() > 1.0:
                    gt_sample = gt_sample / 255.0

                current_iou = iou_score(gt_sample, seg_pred, threshold)
                current_pa = pixel_accuracy(gt_sample, seg_pred, threshold)
                current_dice = dice_loss(gt_sample, seg_pred, threshold)
                current_precision = precision_score(gt_sample, seg_pred, threshold)
                current_recall = recall_score(gt_sample, seg_pred, threshold)

                bg_iou = iou_score_class(gt_sample, seg_pred, threshold, target_class=0)
                fg_iou = iou_score_class(gt_sample, seg_pred, threshold, target_class=1)
                per_class_iou.append([bg_iou, fg_iou])

                total_iou += current_iou
                total_pa += current_pa
                total_dice += current_dice
                total_precision += current_precision
                total_recall += current_recall
                processed_count += 1

                if ii % save_interval == 0 and ii < num_examples_to_save * save_interval:
                    save_example_image_with_pred(
                        frames, pred, gt_sample, seg_pred, ii,
                        current_iou, current_dice, current_precision, current_recall,
                        examples_dir, threshold
                    )

                if (ii + 1) % 100 == 0:
                    print(f"Processed {ii + 1}/{len(testloader)} images... "
                          f"IoU={total_iou / processed_count:.4f} Dice={1 - (total_dice / processed_count):.4f}")

        if processed_count == 0:
            print("No images were processed!")
            continue

        mean_iou = total_iou / processed_count
        mean_pa = total_pa / processed_count
        mean_dice_loss = total_dice / processed_count
        mean_precision = total_precision / processed_count
        mean_recall = total_recall / processed_count

        per_class_iou = np.array(per_class_iou)
        mean_bg_iou = np.mean(per_class_iou[:, 0])
        mean_fg_iou = np.mean(per_class_iou[:, 1])
        mean_iou_classes = (mean_bg_iou + mean_fg_iou) / 2

        f1 = (2 * mean_precision * mean_recall / (mean_precision + mean_recall)) if (mean_precision + mean_recall) > 0 else 0.0
        dice_score = 1 - mean_dice_loss

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Model checkpoint: {model_path}")
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

        # Save results to txt (per threshold)
        results_file = os.path.join(save_model_dir, f"evaluation_results_t{threshold:.1f}.txt")
        with open(results_file, "w") as f:
            f.write("STCNN Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Checkpoint: {model_path}\n")
            f.write(f"Frames: {num_frame}\n")
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


if __name__ == "__main__":
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    main(frame=4, epochs_tag=0, test_thresholds=test_thresholds)
