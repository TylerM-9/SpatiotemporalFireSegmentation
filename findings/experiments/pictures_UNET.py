"""
Testing script for ST-UNet with clean UNet architecture
Adjusted to save examples in vertical 400x710 format
"""

import numpy as np
import os
import cv2
import imageio
from mypath import Path
import torch
from dataloaders import FIRE_dataloader as db
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import clean ST-UNet architecture
from network.UNET_ST import create_stunet_with_attention
from network.joint_pred_seg import FramePredDecoder, FramePredEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# UPDATE THESE PATHS
model_path = "/home/c43n256/STCNN/output/STUNET_UNET_DAVIS_FIRE4/STUNET_UNET_DAVIS_FIRE4-94.pth"
model_name = "STUNET_UNET_FIREDAVIS4"


def load_stunet_model(model_path, num_frame, device):
    """
    Load ST-UNet model from checkpoint
    """
    print(f"Loading ST-UNet model from: {model_path}")

    # Initialize temporal branch components
    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()

    # Create ST-UNet architecture (same as training)
    net = create_stunet_with_attention(
        pred_enc=pred_enc,
        pred_dec=pred_dec,
        num_frame=num_frame,
        n_classes=1
    )

    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)

        print("\n" + "=" * 60)
        print("CHECKPOINT ANALYSIS")
        print("=" * 60)

        # Check checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")

            # Extract state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Loading from 'state_dict' key")

                # Print additional info if available
                if 'epoch' in checkpoint:
                    print(f"Checkpoint epoch: {checkpoint['epoch']}")
                if 'config' in checkpoint:
                    print(f"Model config: {checkpoint['config']}")
            else:
                state_dict = checkpoint
                print("Using checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
            print("Checkpoint is a state_dict")

        # Load weights
        missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"WARNING - Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"WARNING - Unexpected keys: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            print("✓ All weights loaded successfully!")

        print("=" * 60 + "\n")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return None

    net.to(device)
    net.eval()

    return net


def main(frame, epochs, test_thresholds=None, target_example_index=1, selection_metric="mean_iou_classes"):
    """
    Main testing function for ST-UNet model
    1) Evaluate all thresholds
    2) Select best threshold based on selection_metric
    3) Save ONLY the target_example_index at 400x710 for best threshold

    Args:
        frame: Number of input frames
        epochs: Epoch number (for naming)
        test_thresholds: List of thresholds to test
        target_example_index: Which example to save (default: 141)
        selection_metric: Metric to select best threshold (default: "mean_iou_classes")
    """
    if test_thresholds is None:
        test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    num_frame = frame
    num_epochs = epochs
    save_dir = Path.save_root_dir()
    save_model_dir = os.path.join(save_dir, model_name)

    # Load ST-UNet model
    model = load_stunet_model(model_path, num_frame, device)
    if model is None:
        return

    print(f"Testing thresholds: {test_thresholds}")
    print(f"Target example index: {target_example_index}")
    print(f"Selection metric: {selection_metric}")

    # -------------------------
    # 1) Evaluate all thresholds
    # -------------------------
    model.eval()
    threshold_results = []

    # Load test dataset
    test_set = db.FIREDatasetRandom(
        inputRes=(256, 256),
        mode="test",
        num_frame=num_frame
    )

    testloader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=4,
        shuffle=False
    )

    print(f"Total dataset size: {len(testloader)} images")
    print("=" * 60)

    with torch.no_grad():
        for threshold in test_thresholds:
            print(f"\nEvaluating threshold {threshold:.1f} ...")

            # Initialize metric accumulators
            total_iou = 0
            total_pa = 0
            total_dice = 0
            total_precision = 0
            total_recall = 0
            per_class_iou = []
            processed_count = 0

            for ii, sample_batched in enumerate(testloader):
                # Get input and ground truth
                seqs = sample_batched['images'].to(device)
                frames = sample_batched['frame'].to(device)
                gts = sample_batched['seg_gt']

                # Forward pass through ST-UNet
                seg_res, pred = model.forward(seqs, frames)

                # Unpack segmentation result
                if isinstance(seg_res, tuple):
                    seg_logits = seg_res[0]
                    attention_outs = seg_res[1]
                elif isinstance(seg_res, list):
                    seg_logits = seg_res[0]
                else:
                    seg_logits = seg_res

                # Convert logits to probabilities
                seg_pred = torch.sigmoid(seg_logits)
                seg_pred = seg_pred[0, :, :, :].cpu().numpy()

                # Get ground truth
                gt_sample = gts[0, :, :, :].cpu().numpy().transpose([1, 2, 0])
                if gt_sample.max() > 1.0:
                    gt_sample = gt_sample / 255.0

                # Calculate metrics
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

            if processed_count == 0:
                print(f"WARNING: No images processed for threshold {threshold}")
                continue

            # Calculate metrics for this threshold
            mean_iou = total_iou / processed_count
            mean_pa = total_pa / processed_count
            mean_dice = total_dice / processed_count
            mean_precision = total_precision / processed_count
            mean_recall = total_recall / processed_count

            per_class_iou = np.array(per_class_iou)
            mean_bg_iou = float(np.mean(per_class_iou[:, 0]))
            mean_fg_iou = float(np.mean(per_class_iou[:, 1]))
            mean_iou_classes = (mean_bg_iou + mean_fg_iou) / 2.0

            f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (
                    mean_precision + mean_recall) > 0 else 0.0
            dice_score = 1 - mean_dice

            threshold_results.append({
                "threshold": threshold,
                "mean_iou_fg": mean_iou,
                "mean_bg_iou": mean_bg_iou,
                "mean_fg_iou": mean_fg_iou,
                "mean_iou_classes": mean_iou_classes,
                "mean_pa": mean_pa,
                "precision": mean_precision,
                "recall": mean_recall,
                "f1": f1_score,
                "dice_score": dice_score,
                "dice_loss": mean_dice,
                "count": processed_count
            })

            print(f"  -> MeanIoU(all): {mean_iou_classes:.4f} | IoU(FG): {mean_iou:.4f} | "
                  f"PA: {mean_pa:.4f} | F1: {f1_score:.4f} | Dice: {dice_score:.4f}")

    if not threshold_results:
        print("No thresholds evaluated.")
        return

    # -------------------------
    # 2) Select best threshold
    # -------------------------
    valid_keys = {"mean_iou_classes", "mean_iou_fg", "mean_pa", "f1", "dice_score"}
    if selection_metric not in valid_keys:
        print(f"selection_metric '{selection_metric}' not recognized. Defaulting to 'mean_iou_classes'.")
        selection_metric = "mean_iou_classes"

    best_entry = max(threshold_results, key=lambda d: d[selection_metric])
    best_t = best_entry["threshold"]

    print("\n" + "=" * 60)
    print("THRESHOLD SELECTION")
    print("=" * 60)
    for r in threshold_results:
        print(f"t={r['threshold']:.1f} | MeanIoU(all)={r['mean_iou_classes']:.4f} | "
              f"IoU(FG)={r['mean_iou_fg']:.4f} | PA={r['mean_pa']:.4f} | "
              f"F1={r['f1']:.4f} | Dice={r['dice_score']:.4f}")
    print("-" * 60)
    print(f"Best by '{selection_metric}': t={best_t:.1f} (MeanIoU(all)={best_entry['mean_iou_classes']:.4f})")
    print("=" * 60 + "\n")

    # -------------------------
    # 3) Save ONLY target example at best threshold (400x710)
    # -------------------------
    examples_dir = os.path.join(save_model_dir, f"examples_epoch{num_epochs}_best_t{best_t:.1f}")
    os.makedirs(examples_dir, exist_ok=True)
    print(f"Saving ONLY example index {target_example_index} at best threshold {best_t:.1f}")
    print(f"Output dir: {examples_dir}")

    with torch.no_grad():
        for ii, sample_batched in enumerate(testloader):
            if ii != target_example_index:
                continue

            # Get input and ground truth
            seqs = sample_batched['images'].to(device)
            frames = sample_batched['frame'].to(device)
            gts = sample_batched['seg_gt']

            # Forward pass
            seg_res, pred = model.forward(seqs, frames)

            # Unpack segmentation result
            if isinstance(seg_res, tuple):
                seg_logits = seg_res[0]
            elif isinstance(seg_res, list):
                seg_logits = seg_res[0]
            else:
                seg_logits = seg_res

            # Convert to probabilities
            seg_pred = torch.sigmoid(seg_logits)
            seg_pred = seg_pred[0, :, :, :].cpu().numpy()

            # Get ground truth
            gt_sample = gts[0, :, :, :].cpu().numpy().transpose([1, 2, 0])
            if gt_sample.max() > 1.0:
                gt_sample = gt_sample / 255.0

            # Save in vertical format
            save_example_image_vertical_fixed(
                frames, gt_sample, seg_pred, ii, examples_dir, best_t,
                out_w=400, out_h=710,
                pred_h=237, gt_h=237, rgb_h=236
            )
            print(f"✓ Saved sample {ii} at best threshold {best_t:.1f} (400x710)")
            break

    # Save summary file
    summary_path = os.path.join(examples_dir, f"best_threshold_summary_epoch{num_epochs}.txt")
    with open(summary_path, "w") as f:
        f.write("ST-UNet Best Threshold Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Frames: {num_frame}\n")
        f.write(f"Selection metric: {selection_metric}\n")
        f.write(f"Best threshold: {best_t:.3f}\n")
        f.write("\nAll thresholds:\n")
        for r in threshold_results:
            f.write(
                f"t={r['threshold']:.1f} | MeanIoU(all)={r['mean_iou_classes']:.6f} | "
                f"IoU(FG)={r['mean_iou_fg']:.6f} | BGIoU={r['mean_bg_iou']:.6f} | "
                f"PA={r['mean_pa']:.6f} | F1={r['f1']:.6f} | Dice={r['dice_score']:.6f}\n"
            )
    print(f"Saved best-threshold summary to: {summary_path}")


# -------------------------
# Metric functions
# -------------------------
def precision_score(y_true, y_pred, threshold=0.5):
    """Calculate Precision: TP / (TP + FP)"""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    true_positives = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
    false_positives = np.logical_and(y_true_bin == 0, y_pred_bin == 1).sum()
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0


def recall_score(y_true, y_pred, threshold=0.5):
    """Calculate Recall (Sensitivity): TP / (TP + FN)"""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    true_positives = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
    false_negatives = np.logical_and(y_true_bin == 1, y_pred_bin == 0).sum()
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0


def iou_score(y_true, y_pred, threshold=0.5):
    """Calculate IoU score (Intersection over Union)."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()
    return intersection / union if union != 0 else 1.0


def iou_score_class(y_true, y_pred, threshold=0.5, target_class=1):
    """Calculate IoU score for a specific class."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    if target_class == 0:
        y_true_class = (y_true_bin == 0).astype(np.uint8)
        y_pred_class = (y_pred_bin == 0).astype(np.uint8)
    else:
        y_true_class = y_true_bin
        y_pred_class = y_pred_bin
    intersection = np.logical_and(y_true_class, y_pred_class).sum()
    union = np.logical_or(y_true_class, y_pred_class).sum()
    return intersection / union if union != 0 else 1.0


def pixel_accuracy(y_true, y_pred, threshold=0.5):
    """Calculate pixel-wise accuracy."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    correct_pixels = (y_true_bin == y_pred_bin).sum()
    total_pixels = y_true_bin.size
    return correct_pixels / total_pixels if total_pixels > 0 else 1.0


def dice_loss(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """Calculate Dice loss (1 - Dice coefficient)."""
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)
    intersection = (y_true_bin * y_pred_bin).sum()
    dice_coef = (2. * intersection + smooth) / (y_true_bin.sum() + y_pred_bin.sum() + smooth)
    return 1 - dice_coef


# -------------------------
# Visualization (vertical 400x710, no labels)
# -------------------------
def save_example_image_vertical_fixed(input_img, gt_sample, seg_pred, index, save_dir, threshold,
                                      out_w=400, out_h=710,
                                      pred_h=237, gt_h=237, rgb_h=236):
    """
    Create a 400x710 vertical stack:
      Top: prediction mask (B/W, height=pred_h)
      Mid: ground-truth mask (B/W, height=gt_h)
      Bot: original RGB frame (height=rgb_h)
    No labels, no borders.
    """
    assert pred_h + gt_h + rgb_h == out_h, f"Heights must sum to out_h: {pred_h}+{gt_h}+{rgb_h} != {out_h}"

    # --- Get RGB frame in [0,1] HxWx3 ---
    if torch.is_tensor(input_img):
        arr = input_img.detach().cpu().numpy()
    else:
        arr = input_img

    if arr.ndim == 4:   # [B,C,H,W]
        frame = arr[0, -3:, :, :].transpose(1, 2, 0)
    elif arr.ndim == 3:  # [C,H,W]
        frame = arr[-3:, :, :].transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected input shape: {arr.shape}")

    # --- Smart denormalization based on data range ---
    frame_min, frame_max = frame.min(), frame.max()

    if frame_min >= -1.1 and frame_max <= 1.1:
        # Data is in [-1, 1] range
        frame = (frame + 1.0) / 2.0
    elif frame_min >= -0.1 and frame_max <= 1.1:
        # Data is already in [0, 1] range
        frame = frame
    else:
        # Data might be ImageNet normalized or other scheme
        # Try to reverse common ImageNet normalization
        # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        frame = frame * std + mean

    frame = np.clip(frame, 0, 1)

    # --- Masks (0/1) -> 0..255 ---
    gt = np.squeeze(gt_sample).astype(np.float32)
    pred = np.squeeze((seg_pred > threshold).astype(np.float32))

    gt_u8 = (gt * 255.0).astype(np.uint8)
    pred_u8 = (pred * 255.0).astype(np.uint8)

    # --- Resize everything to target widths/heights ---
    pred_res = cv2.resize(pred_u8, (out_w, pred_h), interpolation=cv2.INTER_NEAREST)
    gt_res = cv2.resize(gt_u8, (out_w, gt_h), interpolation=cv2.INTER_NEAREST)

    # Keep original crisp; use AREA for downscale
    frame_u8 = (frame * 255.0).astype(np.uint8)
    frame_res = cv2.resize(frame_u8, (out_w, rgb_h), interpolation=cv2.INTER_AREA)

    # --- Stack (convert masks to 3-ch for consistent stacking) ---
    pred_3 = np.repeat(pred_res[..., None], 3, axis=2)
    gt_3 = np.repeat(gt_res[..., None], 3, axis=2)
    out = np.vstack([pred_3, gt_3, frame_res])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_example_{index:04d}_400x710_t{threshold:.1f}.png")
    imageio.imwrite(save_path, out)


if __name__ == "__main__":
    # Test with multiple thresholds, select best, save one example
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    main(
        frame=4,
        epochs=200,
        test_thresholds=test_thresholds,
        target_example_index=1,
        selection_metric="mean_iou_classes"
    )