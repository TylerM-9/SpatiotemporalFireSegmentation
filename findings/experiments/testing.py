from network.joint_pred_seg import STCNN, FramePredDecoder, FramePredEncoder
import numpy as np
import os
from mypath import Path
import torch
import imageio
from dataloaders import FIRE_dataloader as db
from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils.data import DataLoader, ConcatDataset

from network.MV2_try import SegDecoderJoint, SegEncoder, STCNN2


def main(frame, epochs):
    num_frame = frame
    num_epochs = epochs
    modelName = 'STCNN_frame_FAN' + str(num_frame)

    save_dir = Path.save_root_dir()
    save_model_dir = os.path.join(save_dir, modelName)

    seg_enc = SegEncoder()
    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()
    j_seg_dec = SegDecoderJoint()

    net = STCNN2(pred_enc, seg_enc, pred_dec, j_seg_dec)

    # Fixed: Consistent path construction
    model_path = os.path.join(save_model_dir, modelName + 'Flame-' + str(num_epochs) + '.pth')
    print("Updating weights from: {}".format(model_path))

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        return

    net.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage))

    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
                                              ])

    datasets = []
    # Updated: Use more flexible path construction
    base_image_path = "/home/r56x196/Data/archive-2/Image"
    base_mask_path = "/home/r56x196/Data/archive-2/Mask"

    for i in range(1, 18):
        image_path = os.path.join(base_image_path, f"split_{i}")
        mask_path = os.path.join(base_mask_path, f"split_{i}")

        # Check if paths exist
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            print(f"Warning: Skipping split_{i} - paths don't exist")
            continue

        db_test = db.FIREDatasetGeneral(
            inputRes=(256, 256),
            image_path=image_path,
            mask_path=mask_path,
            transform=composed_transforms,
            num_frame=num_frame
        )
        datasets.append(db_test)

    if not datasets:
        print("Error: No valid datasets found!")
        return

    test_set = db.FIREDatasetRandom(inputRes=(256,256),mode="test", num_frame=num_frame)
    testloader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)

    num_img_test = len(testloader)

    # Fixed: Track actual processed images
    total_iou = 0
    total_pa = 0
    processed_count = 0

    # Create output directory if it doesn't exist
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)

    for ii, sample_batched in enumerate(testloader):
        seqs, frames, gts, pred_gts = sample_batched['images'], sample_batched['frame'], \
            sample_batched['seg_gt'], sample_batched['pred_gt']

        seg_res, pred = net.forward(seqs, frames)

        seg_pred = seg_res[0, :, :, :].data.cpu().numpy()
        seg_pred = 1 / (1 + np.exp(-seg_pred))  # Sigmoid activation (0-1 range)

        # Fixed: Keep ground truth in proper range for evaluation
        gt_sample = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
        # Ensure gt_sample is in 0-1 range (assuming it comes normalized)
        if gt_sample.max() > 1.0:
            gt_sample = gt_sample / 255.0

        pred_gts_sample = pred_gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])

        # Calculate metrics with consistent data ranges (using default threshold 0.5)
        current_iou = iou_score(gt_sample, seg_pred)
        current_pa = pixel_accuracy(gt_sample, seg_pred)

        total_iou += current_iou
        total_pa += current_pa
        processed_count += 1

        print(f"IoU {ii}: {current_iou:.4f}")
        print(f"Pixel Accuracy {ii}: {current_pa:.4f}")

        # Save threshold analysis images (every 10th image to avoid too many files)
        # if ii % 10 == 0:
            # save_threshold_analysis(seg_pred, gt_sample, ii, output_dir)

        # Prepare images for visualization (scale to 0-255 for display)
        seg_pred_display = seg_pred.transpose([1, 2, 0]) * 255
        gt_sample_display = gt_sample * 255

        frame_sample = pred_gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
        frame_sample = inverse_transform(frame_sample) * 255

        # Create 3-channel versions for visualization
        gt_sample3 = np.concatenate([gt_sample_display, gt_sample_display, gt_sample_display], axis=2)
        seg_pred3 = np.concatenate([seg_pred_display, seg_pred_display, seg_pred_display], axis=2)
        samples1 = np.concatenate((seg_pred3, gt_sample3, frame_sample), axis=0)

        # Save result image
        output_path = os.path.join(output_dir, f"test_fire_{ii:04d}_s.png")
        imageio.imwrite(output_path, np.uint8(np.clip(samples1, 0, 255)))


    # Fixed: Use actual processed count for final metrics
    if processed_count > 0:
        print(f"FINAL IoU: {total_iou / processed_count:.4f}")
        print(f"FINAL Pixel Accuracy: {total_pa / processed_count:.4f}")
        print(f"Processed {processed_count} images")

        # Generate overall threshold analysis summary
        generate_threshold_summary(output_dir)
    else:
        print("No images were processed!")


def generate_threshold_summary(output_dir):
    """Generate a summary of optimal thresholds across all analyzed images."""
    threshold_dir = os.path.join(output_dir, "threshold_analysis")

    if not os.path.exists(threshold_dir):
        return

    # Collect all IoU files
    iou_files = [f for f in os.listdir(threshold_dir) if f.startswith("threshold_ious_") and f.endswith(".txt")]

    if not iou_files:
        return

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_ious_all = {thresh: [] for thresh in thresholds}
    best_thresholds = []

    # Parse all IoU files
    for iou_file in sorted(iou_files):
        filepath = os.path.join(threshold_dir, iou_file)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # Parse IoU scores
            for line in lines:
                if '\t' in line and line.replace('\t', '').replace(' ', '').replace('\n', '').replace('.',
                                                                                                      '').isdigit() == False:
                    continue
                try:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        thresh = float(parts[0])
                        iou = float(parts[1])
                        if thresh in threshold_ious_all:
                            threshold_ious_all[thresh].append(iou)
                except:
                    continue

            # Extract best threshold
            for line in lines:
                if line.startswith("Best Threshold:"):
                    try:
                        best_thresh = float(line.split()[2])
                        best_thresholds.append(best_thresh)
                    except:
                        continue
                    break

        except Exception as e:
            print(f"Error reading {iou_file}: {e}")
            continue

    # Generate summary
    summary_path = os.path.join(threshold_dir, "threshold_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("THRESHOLD ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("Average IoU scores by threshold:\n")
        f.write("-" * 30 + "\n")
        f.write("Threshold\tAvg IoU\t\tStd Dev\n")

        avg_ious = {}
        for thresh in thresholds:
            if threshold_ious_all[thresh]:
                avg_iou = np.mean(threshold_ious_all[thresh])
                std_iou = np.std(threshold_ious_all[thresh])
                avg_ious[thresh] = avg_iou
                f.write(f"{thresh:.1f}\t\t{avg_iou:.4f}\t\t{std_iou:.4f}\n")

        if avg_ious:
            best_overall_thresh = max(avg_ious, key=avg_ious.get)
            f.write(
                f"\nBest overall threshold: {best_overall_thresh:.1f} (Avg IoU: {avg_ious[best_overall_thresh]:.4f})\n")

        if best_thresholds:
            f.write(f"\nMost frequently best threshold: {max(set(best_thresholds), key=best_thresholds.count):.1f}\n")
            f.write(f"Average of best thresholds: {np.mean(best_thresholds):.2f}\n")

            # Histogram of best thresholds
            f.write(f"\nDistribution of best thresholds:\n")
            f.write("-" * 30 + "\n")
            for thresh in sorted(set(best_thresholds)):
                count = best_thresholds.count(thresh)
                f.write(f"{thresh:.1f}: {count} images\n")

    print(f"Threshold analysis summary saved to: {summary_path}")


def save_threshold_analysis(seg_pred, gt_sample, image_idx, output_dir):
    """Save threshold analysis images showing binary masks at different thresholds."""
    # Test different threshold values
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Create threshold analysis directory
    threshold_dir = os.path.join(output_dir, "threshold_analysis")
    os.makedirs(threshold_dir, exist_ok=True)

    # Prepare ground truth for comparison (ensure it's binary)
    gt_binary = (gt_sample > 0.5).astype(np.uint8)
    gt_display = np.squeeze(gt_binary) * 255

    # Original prediction (grayscale)
    pred_grayscale = np.squeeze(seg_pred) * 255

    # Create a grid showing: Original | GT | Thresholds
    rows = []

    # First row: Original prediction and ground truth
    first_row = np.concatenate([
        pred_grayscale,
        gt_display,
        np.full_like(pred_grayscale, 128)  # Gray separator
    ], axis=1)
    rows.append(first_row)

    # Add separator row
    separator = np.full((20, first_row.shape[1]), 200)
    rows.append(separator)

    # Create rows of thresholded images (3 per row)
    threshold_images = []
    threshold_ious = []

    for thresh in thresholds:
        # Apply threshold
        thresh_binary = (seg_pred > thresh).astype(np.uint8)
        thresh_display = np.squeeze(thresh_binary) * 255

        # Calculate IoU for this threshold
        iou = iou_score(gt_sample, seg_pred, threshold=thresh)
        threshold_ious.append(iou)

        threshold_images.append((thresh_display, thresh, iou))

    # Arrange threshold images in rows of 3
    for i in range(0, len(threshold_images), 3):
        row_images = threshold_images[i:i + 3]

        # Pad with empty if needed
        while len(row_images) < 3:
            row_images.append((np.full_like(pred_grayscale, 0), 0.0, 0.0))

        # Create row
        row = np.concatenate([img[0] for img in row_images], axis=1)
        rows.append(row)

        # Add text info (create a text row)
        text_height = 30
        text_row = np.full((text_height, row.shape[1]), 255)  # White background
        rows.append(text_row)

    # Combine all rows
    final_image = np.concatenate(rows, axis=0)

    # Save the threshold analysis image
    analysis_path = os.path.join(threshold_dir, f"threshold_analysis_{image_idx:04d}.png")
    imageio.imwrite(analysis_path, np.uint8(final_image))

    # Save IoU scores to text file
    iou_path = os.path.join(threshold_dir, f"threshold_ious_{image_idx:04d}.txt")
    with open(iou_path, 'w') as f:
        f.write(f"Threshold Analysis for Image {image_idx}\n")
        f.write("=" * 40 + "\n")
        f.write("Threshold\tIoU Score\n")
        f.write("-" * 20 + "\n")
        for thresh, iou in zip(thresholds, threshold_ious):
            f.write(f"{thresh:.1f}\t\t{iou:.4f}\n")

        # Find best threshold
        best_idx = np.argmax(threshold_ious)
        best_thresh = thresholds[best_idx]
        best_iou = threshold_ious[best_idx]
        f.write(f"\nBest Threshold: {best_thresh:.1f} (IoU: {best_iou:.4f})\n")

    print(f"Threshold analysis saved for image {image_idx}")
    print(f"Best threshold: {best_thresh:.1f} with IoU: {best_iou:.4f}")


def iou_score(y_true, y_pred, threshold=0.5):
    """Calculate IoU score with proper handling of edge cases."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)  # Ensure binary

    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)

    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()

    return intersection / union if union != 0 else 1.0  # Return 1.0 if both are empty


def inverse_transform(images):
    """Inverse transform to convert from [-1,1] to [0,1] range."""
    return (images + 1.) / 2.


def pixel_accuracy(y_true, y_pred, threshold=0.5):
    """Calculate pixel accuracy with proper data handling."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)  # Ensure binary

    y_true_bin = np.squeeze(y_true_bin)
    y_pred_bin = np.squeeze(y_pred_bin)

    correct_pixels = (y_true_bin == y_pred_bin).sum()
    total_pixels = y_true_bin.size

    return correct_pixels / total_pixels if total_pixels > 0 else 1.0


if __name__ == "__main__":
    main(4, 174)