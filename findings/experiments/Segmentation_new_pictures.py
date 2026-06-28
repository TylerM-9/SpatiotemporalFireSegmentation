import numpy as np
import os
import cv2
import imageio
from mypath import Path
import torch
from dataloaders import FIRE_dataloader as db
from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import only the segmentation model
from network.joint_pred_seg import SegBranch, SegDecoder, SegEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "/home/r56x196/STCNN/output/Seg_Branch_FIRE_with_validation/Seg_Branch_FIRE_with_validation_epoch-200.pth"
model_name = "Seg_Branch_FIRE_with_validation"


# --- Fallback dict-aware ToTensor in case tr.ToTensor() doesn't exist or isn't dict-aware ---
class ToTensorDict:
    """
    Safely converts numpy/PIL fields in a sample dict to torch tensors.
    Expects keys like 'images', 'seg_gt', etc. Leaves non-array entries untouched.
    """
    def __call__(self, sample):
        out = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                out[k] = v
            elif hasattr(v, "numpy"):  # already something tensor-like
                out[k] = torch.as_tensor(v)
            elif isinstance(v, np.ndarray):
                # If image is HWC or CHW, just convert; no normalization here.
                out[k] = torch.from_numpy(v)
            else:
                # Some loaders keep PIL for images; convert those too
                try:
                    from torchvision.transforms.functional import to_tensor
                    if "PIL" in str(type(v)):
                        out[k] = to_tensor(v)
                    else:
                        out[k] = v
                except Exception:
                    out[k] = v
        return out


def _dict_aware_transform():
    """
    Prefer your project's custom dict-aware transform if available.
    Fallback to ToTensorDict() if not.
    """
    # Many codebases define tr.ToTensor() that handles dict samples.
    tfm = None
    if hasattr(tr, "ToTensor"):
        try:
            tfm = tr.ToTensor()
        except Exception:
            tfm = None

    if tfm is None:
        tfm = ToTensorDict()

    # If you also have dict-aware Resize/Normalize in custom_transforms, add them here.
    # Keep it minimal to avoid mismatching your training preprocessing.
    return tfm


def main(epochs, test_thresholds=None, target_example_index=141, selection_metric="mean_iou_classes"):
    """
    1) Evaluate thresholds
    2) Pick best by `selection_metric`
    3) Save ONLY sample #target_example_index for the best threshold at 400x710
    """
    if test_thresholds is None:
        test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # -------------------------
    # Setup save dirs and model
    # -------------------------
    save_root = Path.save_root_dir()
    save_model_dir = os.path.join(save_root, model_name)

    encoder = SegEncoder()
    decoder = SegDecoder()
    seg_net = SegBranch(net_enc=encoder, net_dec=decoder).to(device)

    # -------------------------
    # Load weights
    # -------------------------
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    seg_keys = [k for k in state_dict.keys() if k.startswith("seg.")]
    if seg_keys:
        print("Extracting segmentation weights (removing 'seg.' prefix)...")
        seg_weights = {k.replace("seg.", ""): v for k, v in state_dict.items() if k.startswith("seg.")}
        missing_keys, unexpected_keys = seg_net.load_state_dict(seg_weights, strict=False)
    else:
        print("Loading weights directly (no 'seg.' prefix found)...")
        missing_keys, unexpected_keys = seg_net.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"WARNING - Missing keys ({len(missing_keys)}):")
        for k in missing_keys[:10]:
            print(f"  - {k}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    if unexpected_keys:
        print(f"WARNING - Unexpected keys ({len(unexpected_keys)}):")
        for k in unexpected_keys[:10]:
            print(f"  - {k}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")

    # -------------------------
    # Data (single frame test) — pass dict-aware transform
    # -------------------------
    dict_tfm = _dict_aware_transform()

    test_set = db.FIREDataset(
        inputRes=(256, 256),
        mode="test",
        num_frame=1,               # segmentation-only
        transform=dict_tfm         # <<< IMPORTANT: dict-aware transform
    )
    testloader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=4,
        shuffle=False  # keep indices stable; index 141 is deterministic
    )
    print(f"Total dataset size (batches): {len(testloader)} images")
    print("=" * 60)

    # -------------------------
    # 1) Evaluate thresholds (no images saved here)
    # -------------------------
    seg_net.eval()
    threshold_results = []  # list of dicts per threshold

    with torch.no_grad():
        for threshold in test_thresholds:
            print(f"Evaluating threshold {threshold:.1f} ...")
            total_iou_fg = 0.0
            total_pa = 0.0
            total_dice_loss = 0.0
            total_precision = 0.0
            total_recall = 0.0
            per_class_iou = []
            processed_count = 0

            for ii, sample_batched in enumerate(testloader):
                # sample_batched is a dict with tensors thanks to dict_tfm
                seqs = sample_batched["images"].to(device)
                gts = sample_batched["seg_gt"]

                seg_res = seg_net.forward(seqs)
                seg_output = seg_res[0] if isinstance(seg_res, (list, tuple)) else seg_res

                # (C,H,W) for a single-item batch
                seg_pred = seg_output[0, :, :, :].data.cpu().numpy()
                seg_pred = 1 / (1 + np.exp(-seg_pred))

                gt_sample = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
                if gt_sample.max() > 1.0:
                    gt_sample = gt_sample / 255.0

                current_iou_fg = iou_score(gt_sample, seg_pred, threshold)
                current_pa = pixel_accuracy(gt_sample, seg_pred, threshold)
                current_dice = dice_loss(gt_sample, seg_pred, threshold)
                current_precision = precision_score(gt_sample, seg_pred, threshold)
                current_recall = recall_score(gt_sample, seg_pred, threshold)

                bg_iou = iou_score_class(gt_sample, seg_pred, threshold, target_class=0)
                fg_iou = iou_score_class(gt_sample, seg_pred, threshold, target_class=1)
                per_class_iou.append([bg_iou, fg_iou])

                total_iou_fg += current_iou_fg
                total_pa += current_pa
                total_dice_loss += current_dice
                total_precision += current_precision
                total_recall += current_recall
                processed_count += 1

            if processed_count == 0:
                print("WARNING: No images processed for threshold", threshold)
                continue

            mean_iou_fg = total_iou_fg / processed_count
            mean_pa = total_pa / processed_count
            mean_dice_loss = total_dice_loss / processed_count
            mean_precision = total_precision / processed_count
            mean_recall = total_recall / processed_count
            per_class_iou = np.array(per_class_iou)
            mean_bg_iou = float(np.mean(per_class_iou[:, 0]))
            mean_fg_iou = float(np.mean(per_class_iou[:, 1]))
            mean_iou_classes = (mean_bg_iou + mean_fg_iou) / 2.0
            f1_score_val = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0.0
            dice_score_val = 1.0 - mean_dice_loss

            threshold_results.append({
                "threshold": threshold,
                "mean_iou_fg": mean_iou_fg,
                "mean_bg_iou": mean_bg_iou,
                "mean_fg_iou": mean_fg_iou,
                "mean_iou_classes": mean_iou_classes,
                "mean_pa": mean_pa,
                "precision": mean_precision,
                "recall": mean_recall,
                "f1": f1_score_val,
                "dice_score": dice_score_val,
                "dice_loss": mean_dice_loss,
                "count": processed_count
            })

            print(f"  -> MeanIoU(all): {mean_iou_classes:.4f} | IoU(FG): {mean_iou_fg:.4f} | PA: {mean_pa:.4f} | F1: {f1_score_val:.4f} | Dice: {dice_score_val:.4f}")

    if not threshold_results:
        print("No thresholds evaluated.")
        return

    # -------------------------
    # 2) Choose best threshold
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
        print(f"t={r['threshold']:.1f} | MeanIoU(all)={r['mean_iou_classes']:.4f} | IoU(FG)={r['mean_iou_fg']:.4f} | PA={r['mean_pa']:.4f} | F1={r['f1']:.4f} | Dice={r['dice_score']:.4f}")
    print("-" * 60)
    print(f"Best by '{selection_metric}': t={best_t:.1f} (MeanIoU(all)={best_entry['mean_iou_classes']:.4f})")
    print("=" * 60 + "\n")

    # -------------------------
    # 3) Save ONLY the target example for best threshold at 400x710
    # -------------------------
    examples_dir = os.path.join(save_model_dir, f"examples_epoch{epochs}_best_t{best_t:.1f}")
    os.makedirs(examples_dir, exist_ok=True)
    print(f"Saving ONLY the forced example index {target_example_index} at best threshold {best_t:.1f}")
    print(f"Output dir: {examples_dir}")

    with torch.no_grad():
        for ii, sample_batched in enumerate(testloader):
            if ii != target_example_index:
                continue

            seqs = sample_batched["images"].to(device)
            gts = sample_batched["seg_gt"]

            seg_res = seg_net.forward(seqs)
            seg_output = seg_res[0] if isinstance(seg_res, (list, tuple)) else seg_res

            seg_pred = seg_output[0, :, :, :].data.cpu().numpy()
            seg_pred = 1 / (1 + np.exp(-seg_pred))

            gt_sample = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
            if gt_sample.max() > 1.0:
                gt_sample = gt_sample / 255.0

            save_example_image_vertical_fixed(
                seqs, gt_sample, seg_pred, ii, examples_dir, best_t,
                out_w=400, out_h=710,  # Target output size
                pred_h=237, gt_h=237, rgb_h=236  # Heights sum to 710
            )
            print(f"✓ Saved sample {ii} at best threshold {best_t:.1f} (400x710)")
            break  # save only that single forced example

    summary_path = os.path.join(examples_dir, f"best_threshold_summary_epoch{epochs}.txt")
    with open(summary_path, "w") as f:
        f.write("Best Threshold Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Epochs: {epochs}\n")
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
# Metric helpers
# -------------------------
def precision_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin); y_pred_bin = np.squeeze(y_pred_bin)
    tp = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
    fp = np.logical_and(y_true_bin == 0, y_pred_bin == 1).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin); y_pred_bin = np.squeeze(y_pred_bin)
    tp = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
    fn = np.logical_and(y_true_bin == 1, y_pred_bin == 0).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def iou_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin); y_pred_bin = np.squeeze(y_pred_bin)
    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()
    return intersection / union if union != 0 else 1.0


def iou_score_class(y_true, y_pred, threshold=0.5, target_class=1):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin); y_pred_bin = np.squeeze(y_pred_bin)
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
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_true_bin = np.squeeze(y_true_bin); y_pred_bin = np.squeeze(y_pred_bin)
    correct = (y_true_bin == y_pred_bin).sum()
    total = y_true_bin.size
    return correct / total if total > 0 else 1.0


def dice_loss(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_true_bin = np.squeeze(y_true_bin); y_pred_bin = np.squeeze(y_pred_bin)
    intersection = (y_true_bin * y_pred_bin).sum()
    dice_coef = (2.0 * intersection + smooth) / (y_true_bin.sum() + y_pred_bin.sum() + smooth)
    return 1.0 - dice_coef


# -------------------------
# Visualization (vertical, 400x710, no labels)
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

    # --- get RGB frame in [0,1] HxWx3 ---
    if torch.is_tensor(input_img):
        arr = input_img.detach().cpu().numpy()
    else:
        arr = input_img
    if arr.ndim == 4:   # [B,C,H,W]
        frame = arr[0, -3:, :, :].transpose(1, 2, 0)
    elif arr.ndim == 3: # [C,H,W]
        frame = arr[-3:, :, :].transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected input shape: {arr.shape}")

    frame = (frame + 1.0) / 2.0
    frame = np.clip(frame, 0, 1)

    # --- masks (0/1) -> 0..255 ---
    gt   = np.squeeze(gt_sample).astype(np.float32)
    pred = np.squeeze((seg_pred > threshold).astype(np.float32))

    gt_u8   = (gt   * 255.0).astype(np.uint8)
    pred_u8 = (pred * 255.0).astype(np.uint8)

    # --- resize everything to target widths/heights ---
    pred_res = cv2.resize(pred_u8, (out_w, pred_h), interpolation=cv2.INTER_NEAREST)
    gt_res   = cv2.resize(gt_u8,   (out_w, gt_h),   interpolation=cv2.INTER_NEAREST)

    # keep original crisp; use AREA for downscale
    frame_u8 = (frame * 255.0).astype(np.uint8)
    frame_res = cv2.resize(frame_u8, (out_w, rgb_h), interpolation=cv2.INTER_AREA)

    # --- stack (convert masks to 3-ch for consistent stacking) ---
    pred_3 = np.repeat(pred_res[..., None], 3, axis=2)
    gt_3   = np.repeat(gt_res[...,   None], 3, axis=2)
    out    = np.vstack([pred_3, gt_3, frame_res])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_example_{index:04d}_400x710_t{threshold:.1f}.png")
    imageio.imwrite(save_path, out)


if __name__ == "__main__":
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    main(
        epochs=99,
        test_thresholds=test_thresholds,
        target_example_index=141,
        selection_metric="mean_iou_classes"
    )