import numpy as np
import os
import cv2
import imageio
import numpy as np
import os
import torch
from mypath import Path
import torch
from dataloaders import FIRE_dataloader as db
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from network.joint_pred_seg import FramePredDecoder, FramePredEncoder, SegEncoder, JointSegDecoder, STCNN

# -------------------------
# Dict-aware transform (lives in THIS file)
# -------------------------
class DictToTensor:
    """
    Convert the dataset's sample-dict field-by-field so nothing tries to
    run torchvision.transforms.ToTensor() on the whole dict.

    Assumes the dataset has already done any value scaling it wants
    (e.g., images to [-1,1], frame to ImageNet mean/std, etc.).
    We only convert numpy -> torch and enforce channel-first shape.
    """
    def __call__(self, sample):
        out = {}

        # images: H x W x (3*num_frame) -> [C,H,W] float32
        im = sample["images"]
        if torch.is_tensor(im):
            out["images"] = im
        else:
            v = np.asarray(im, dtype=np.float32)
            if v.ndim == 3 and v.shape[-1] != 1:
                v = torch.from_numpy(v).permute(2, 0, 1).contiguous()
            else:
                v = torch.from_numpy(v).contiguous()
            out["images"] = v

        # frame: H x W x 3 -> [3,H,W] float32
        fr = sample["frame"]
        if torch.is_tensor(fr):
            out["frame"] = fr
        else:
            v = np.asarray(fr, dtype=np.float32)
            if v.ndim == 3 and v.shape[-1] != 1:
                v = torch.from_numpy(v).permute(2, 0, 1).contiguous()
            else:
                v = torch.from_numpy(v).contiguous()
            out["frame"] = v

        # seg_gt: [H,W] or [H,W,1] -> [1,H,W] float32 in {0,1}
        gt = sample["seg_gt"]
        if torch.is_tensor(gt):
            gtt = gt
            if gtt.ndim == 2:
                gtt = gtt.unsqueeze(0)
            elif gtt.ndim == 3 and gtt.shape[0] != 1 and gtt.shape[-1] == 1:
                gtt = gtt.permute(2, 0, 1)
            out["seg_gt"] = gtt.float()
        else:
            v = np.asarray(gt, dtype=np.float32)
            if v.ndim == 2:
                v = v[None, ...]
            elif v.ndim == 3 and v.shape[-1] == 1:
                v = np.transpose(v, (2, 0, 1))
            out["seg_gt"] = torch.from_numpy(v.astype(np.float32))

        # pred_gt: H x W x 3 -> [3,H,W] (kept for your visualization logic)
        pg = sample.get("pred_gt", None)
        if pg is not None:
            if torch.is_tensor(pg):
                out["pred_gt"] = pg
            else:
                v = np.asarray(pg, dtype=np.float32)
                if v.ndim == 3 and v.shape[-1] != 1:
                    v = torch.from_numpy(v).permute(2, 0, 1).contiguous()
                else:
                    v = torch.from_numpy(v).contiguous()
                out["pred_gt"] = v

        # Keep any other keys unchanged
        for k, v in sample.items():
            if k not in out:
                out[k] = v
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "/home/r56x196/STCNN/output/STCNN_frame_NO_DAVIS4/STCNN_frame_NO_DAVIS4Flame-149.pth"
model_name = "STCNN_frame_NO_DAVIS4"


def main(frame, epochs, test_thresholds=None, target_example_index=141, selection_metric="mean_iou_classes"):
    """
    1) Evaluate STCNN across thresholds
    2) Choose best threshold by `selection_metric`
    3) Save ONLY sample `target_example_index` as a clean vertical stack (Pred / GT / Frame)
    """
    if test_thresholds is None:
        test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    num_frame = frame
    num_epochs = epochs

    save_root = Path.save_root_dir()
    save_model_dir = os.path.join(save_root, model_name)
    os.makedirs(save_model_dir, exist_ok=True)

    # -------------------------
    # Build & load model
    # -------------------------
    seg_enc = SegEncoder()
    j_seg_dec = JointSegDecoder()
    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()
    net = STCNN(pred_enc, seg_enc, pred_dec, j_seg_dec).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Resuming from: {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # -------------------------
    # Data (keep order stable)
    # -------------------------
    # Pass our dict-aware transform so the dataset never tries ToTensor() on a dict.
    test_set = db.FIREDataset(
        inputRes=(256, 256),
        mode="test",
        num_frame=num_frame,
        transform=DictToTensor()
    )
    testloader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Total dataset size: {len(testloader)} images")
    print("=" * 60)

    # -------------------------
    # 1) Evaluate thresholds
    # -------------------------
    threshold_results = []
    with torch.no_grad():
        for threshold in test_thresholds:
            print(f"Evaluating threshold {threshold:.1f} ...")
            total_iou = 0.0
            total_pa = 0.0
            total_dice = 0.0
            total_precision = 0.0
            total_recall = 0.0
            per_class_iou = []
            processed_count = 0

            for ii, sample_batched in enumerate(testloader):
                seqs = sample_batched['images'].to(device)
                frames = sample_batched['frame'].to(device)
                gts = sample_batched['seg_gt']

                seg_res, _ = net.forward(seqs, frames)
                seg_output = seg_res[0] if isinstance(seg_res, list) else seg_res

                # (C,H,W) for a single-item batch
                seg_pred = seg_output[0, :, :, :].data.cpu().numpy()
                seg_pred = 1.0 / (1.0 + np.exp(-seg_pred))

                gt_sample = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
                if gt_sample.max() > 1.0:
                    gt_sample = gt_sample / 255.0

                # Metrics
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
                print("WARNING: No images processed.")
                continue

            mean_iou_fg = total_iou / processed_count
            mean_pa = total_pa / processed_count
            mean_dice_loss = total_dice / processed_count
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
    # 3) Save ONLY the target example (#141 by default) at best threshold
    # -------------------------
    examples_dir = os.path.join(save_model_dir, f"examples_epoch{num_epochs}_best_t{best_t:.1f}")
    os.makedirs(examples_dir, exist_ok=True)
    print(f"Saving ONLY sample index {target_example_index} at best threshold {best_t:.1f}")
    print(f"Output dir: {examples_dir}")

    with torch.no_grad():
        for ii, sample_batched in enumerate(testloader):
            if ii != target_example_index:
                continue

            seqs = sample_batched['images'].to(device)
            frames = sample_batched['frame'].to(device)
            gts = sample_batched['seg_gt']

            seg_res, _ = net.forward(seqs, frames)
            seg_output = seg_res[0] if isinstance(seg_res, list) else seg_res

            seg_pred = seg_output[0, :, :, :].data.cpu().numpy()
            seg_pred = 1.0 / (1.0 + np.exp(-seg_pred))

            gt_sample = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
            if gt_sample.max() > 1.0:
                gt_sample = gt_sample / 255.0

            save_example_image_vertical_fixed(seqs, gt_sample, seg_pred, ii, examples_dir, best_t)
            print(f"✓ Saved sample {ii} at best threshold {best_t:.1f}")
            break  # only that single forced example

    # Write summary text
    summary_path = os.path.join(examples_dir, f"best_threshold_summary_epoch{epochs}.txt")
    with open(summary_path, "w") as f:
        f.write("Best Threshold Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Frames: {num_frame}\n")
        f.write(f"Selection metric: {selection_metric}\n")
        f.write(f"Best threshold: {best_t:.3f}\n\n")
        f.write("All thresholds:\n")
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
    y_true_bin = (y_pred > threshold).astype(np.float32)  # NOTE: original had y_true_bin = (y_true > threshold)
    # Fix: use y_true, not y_pred
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_true_bin = np.squeeze(y_true_bin); y_pred_bin = np.squeeze(y_pred_bin)
    intersection = (y_true_bin * y_pred_bin).sum()
    dice_coef = (2.0 * intersection + smooth) / (y_true_bin.sum() + y_pred_bin.sum() + smooth)
    return 1.0 - dice_coef


# -------------------------
# Visualization (vertical, rectangular, no labels)
# -------------------------
def save_example_image_vertical_fixed(input_img, gt_sample, seg_pred, index, save_dir, threshold,
                                      out_w=400, out_h=710,
                                      pred_h=237, gt_h=237, rgb_h=236):
    """
    Create a 300x256 (default) vertical stack:
      Top: prediction mask (B/W, height=pred_h)
      Mid: ground-truth mask (B/W, height=gt_h)
      Bot: original RGB frame (height=rgb_h)
    No labels, no borders.
    """
    assert pred_h + gt_h + rgb_h == out_h, "Heights must sum to out_h"

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
    save_path = os.path.join(save_dir, f"best_example_{index:04d}_300x256_t{threshold:.1f}.png")
    imageio.imwrite(save_path, out)


if __name__ == "__main__":
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    main(
        frame=4,
        epochs=99,
        test_thresholds=test_thresholds,
        target_example_index=141,          # <- change this if you want a different sample
        selection_metric="mean_iou_classes" # alternatives: "mean_iou_fg", "mean_pa", "f1", "dice_score"
    )
