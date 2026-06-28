"""
Testing script for STCNN Joint Prediction & Segmentation Model
Matches the multi-round training architecture
"""

import numpy as np
import os
from mypath import Path
import torch
from dataloaders import FIRE_dataloader as db
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Import STCNN architecture components
from network.joint_pred_seg import (
    STCNN,
    FramePredEncoder,
    FramePredDecoder,
    SegEncoder,
    JointSegDecoder
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# UPDATE THESE PATHS - point to your trained model
model_path = "/home/r56x196/STCNN/output/STCNN_frame_NO_DAVIS4_round2/STCNN_frame_NO_DAVIS4_round2_best.pth"
model_name = "STCNN_JointPredSeg"


def load_stcnn_model(model_path, num_frame, device):
    """
    Load STCNN model from checkpoint

    Args:
        model_path: Path to the saved model checkpoint
        num_frame: Number of input frames (e.g., 4)
        device: torch device (cuda or cpu)

    Returns:
        Loaded STCNN model
    """
    print(f"Loading STCNN model from: {model_path}")

    # Initialize model components (same as training)
    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()
    seg_enc = SegEncoder()
    seg_dec = JointSegDecoder()

    # Create STCNN model
    net = STCNN(pred_enc, pred_dec, seg_enc, seg_dec)

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
                if 'best_val_loss' in checkpoint:
                    print(f"Best validation loss: {checkpoint['best_val_loss']}")
            else:
                state_dict = checkpoint
                print("Using checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
            print("Checkpoint is a state_dict")

        # Load weights
        missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"WARNING - Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"WARNING - Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        if not missing_keys and not unexpected_keys:
            print("✓ All weights loaded successfully!")

        print("=" * 60 + "\n")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return None

    net.to(device)
    net.eval()

    return net


def main(frame, epochs, test_thresholds=None, model_round=0):
    """
    Main testing function for STCNN model

    Args:
        frame: Number of input frames
        epochs: Epoch number (for naming)
        test_thresholds: List of thresholds to test
        model_round: Which training round to test (0, 1, or 2)
    """
    if test_thresholds is None:
        test_thresholds = [0.5]  # Default threshold

    num_frame = frame
    num_epochs = epochs
    save_dir = Path.save_root_dir()

    # Adjust model name based on round
    model_dir_name = f"{model_name}_round{model_round}"
    save_model_dir = os.path.join(save_dir, model_dir_name)
    os.makedirs(save_model_dir, exist_ok=True)

    # Load STCNN model
    model = load_stcnn_model(model_path, num_frame, device)
    if model is None:
        return

    print(f"Testing model from round {model_round}")
    print(f"Testing thresholds: {test_thresholds}")

    for threshold in test_thresholds:
        print(f"\n{'=' * 60}")
        print(f"Testing with threshold: {threshold}")
        print(f"{'=' * 60}")

        model.eval()

        # Load test dataset
        test_set = db.FIREDataset(
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

        # Create examples directory
        examples_dir = os.path.join(save_model_dir, f"examples_epoch{num_epochs}_t{threshold:.1f}")
        os.makedirs(examples_dir, exist_ok=True)
        print(f"Saving example images to: {examples_dir}")

        # Initialize metric accumulators
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
                # Get input and ground truth
                seqs = sample_batched['images'].to(device)
                frames = sample_batched['frame'].to(device)
                gts = sample_batched['seg_gt']

                # Forward pass through STCNN
                # Returns: (seg_res, pred_frame)
                # seg_res can be a list of multi-scale outputs
                seg_res, pred = model.forward(seqs, frames)

                # Handle multi-scale segmentation outputs
                if isinstance(seg_res, list):
                    # Use the finest scale (last element) for evaluation
                    seg_logits = seg_res[-1]
                else:
                    seg_logits = seg_res

                # Convert logits to probabilities (apply sigmoid)
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

                # Save example images
                if ii % save_interval == 0 and ii < num_examples_to_save * save_interval:
                    # Also save the predicted frame
                    save_example_image_with_pred(frames, pred, gt_sample, seg_pred, ii,
                                                 current_iou, current_dice, current_precision,
                                                 current_recall, examples_dir, threshold)

                # Progress update
                if (ii + 1) % 100 == 0:
                    print(f"Processed {ii + 1}/{len(testloader)} images...")
                    print(f"  Running avg - IoU: {total_iou / processed_count:.4f}, "
                          f"PA: {total_pa / processed_count:.4f}, "
                          f"Dice: {total_dice / processed_count:.4f}")

        # Calculate final metrics
        if processed_count > 0:
            mean_iou = total_iou / processed_count
            mean_pa = total_pa / processed_count
            mean_dice = total_dice / processed_count
            mean_precision = total_precision / processed_count
            mean_recall = total_recall / processed_count

            per_class_iou = np.array(per_class_iou)
            mean_bg_iou = np.mean(per_class_iou[:, 0])
            mean_fg_iou = np.mean(per_class_iou[:, 1])
            mean_iou_classes = (mean_bg_iou + mean_fg_iou) / 2

            f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (
                                                                                                      mean_precision + mean_recall) > 0 else 0

            # Print results
            print("\n" + "=" * 60)
            print("FINAL RESULTS")
            print("=" * 60)
            print(f"Model: {model_name} (Round {model_round})")
            print(f"Architecture: STCNN (Joint Prediction + Segmentation)")
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
            print(f"F1 Score:               {f1_score:.4f}")
            print(f"Dice Score:             {1 - mean_dice:.4f}")
            print(f"Dice Loss:              {mean_dice:.4f}")
            print("=" * 60)

            # Save results to file
            results_file = os.path.join(save_model_dir, f"evaluation_results_t{threshold:.1f}.txt")
            with open(results_file, 'w') as f:
                f.write("STCNN Joint Prediction & Segmentation Model Evaluation Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Round: {model_round}\n")
                f.write(f"Architecture: STCNN (FramePred + Seg with JointDecoder)\n")
                f.write(f"Epochs: {num_epochs}\n")
                f.write(f"Frames: {num_frame}\n")
                f.write(f"Threshold: {threshold}\n")
                f.write(f"Dataset Size: {processed_count} images\n")
                f.write("-" * 50 + "\n")
                f.write(f"IoU (Foreground):       {mean_iou:.6f}\n")
                f.write(f"Mean IoU (All Classes): {mean_iou_classes:.6f}\n")
                f.write(f"  - Background IoU:     {mean_bg_iou:.6f}\n")
                f.write(f"  - Foreground IoU:     {mean_fg_iou:.6f}\n")
                f.write(f"Mean Pixel Accuracy:    {mean_pa:.6f}\n")
                f.write(f"Precision:              {mean_precision:.6f}\n")
                f.write(f"Recall:                 {mean_recall:.6f}\n")
                f.write(f"F1 Score:               {f1_score:.6f}\n")
                f.write(f"Dice Score:             {1 - mean_dice:.6f}\n")
                f.write(f"Dice Loss:              {mean_dice:.6f}\n")

            print(f"Results saved to: {results_file}")

            # Save comparison grid
            save_comparison_grid(examples_dir, num_examples=9)
        else:
            print("No images were processed!")


# Metric functions
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


def save_example_image_with_pred(input_img, pred_frame, gt_sample, seg_pred, index,
                                 iou, dice, precision, recall, save_dir, threshold):
    """Save a visualization with input, predicted frame, ground truth, and segmentation prediction."""
    # Process input frame
    if torch.is_tensor(input_img):
        input_numpy = input_img.cpu().numpy()
    else:
        input_numpy = input_img

    if len(input_numpy.shape) == 4:
        input_frame = input_numpy[0, :, :, :].transpose(1, 2, 0)
    elif len(input_numpy.shape) == 3:
        input_frame = input_numpy.transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected input shape: {input_numpy.shape}")

    # Denormalize input frame
    input_frame = (input_frame + 1.0) / 2.0
    input_frame = np.clip(input_frame, 0, 1)

    # Process predicted frame
    if torch.is_tensor(pred_frame):
        pred_numpy = pred_frame.cpu().numpy()
    else:
        pred_numpy = pred_frame

    if len(pred_numpy.shape) == 4:
        pred_display = pred_numpy[0, :, :, :].transpose(1, 2, 0)
    elif len(pred_numpy.shape) == 3:
        pred_display = pred_numpy.transpose(1, 2, 0)
    else:
        pred_display = pred_numpy

    # Denormalize predicted frame (tanh output: [-1, 1])
    pred_display = (pred_display + 1.0) / 2.0
    pred_display = np.clip(pred_display, 0, 1)

    # Process segmentation
    gt_display = np.squeeze(gt_sample)
    pred_binary = (seg_pred > threshold).astype(np.float32)
    pred_seg_display = np.squeeze(pred_binary)

    # Create 2x2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Row 1: Frame prediction
    axes[0, 0].imshow(input_frame)
    axes[0, 0].set_title('Input Frame', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_display)
    axes[0, 1].set_title('Predicted Next Frame', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Row 2: Segmentation
    axes[1, 0].imshow(input_frame)
    axes[1, 0].imshow(gt_display, alpha=0.5, cmap='jet')
    axes[1, 0].set_title('Ground Truth Segmentation', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(input_frame)
    axes[1, 1].imshow(pred_seg_display, alpha=0.5, cmap='jet')
    axes[1, 1].set_title(
        f'Predicted Segmentation\nIoU: {iou:.4f} | Dice: {1 - dice:.4f}\nP: {precision:.4f} | R: {recall:.4f}',
        fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    fig.suptitle(f'STCNN Sample {index} - Threshold: {threshold}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    save_path = os.path.join(save_dir, f'example_{index:04d}_iou{iou:.3f}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_grid(examples_dir, num_examples=9):
    """Create a grid of saved examples for quick overview."""
    import glob
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


def test_all_rounds(frame=4, epochs=100, test_thresholds=None):
    """Test models from all 3 training rounds"""
    if test_thresholds is None:
        test_thresholds = [0.5]

    save_dir = Path.save_root_dir()

    for round_num in range(3):
        print(f"\n{'=' * 80}")
        print(f"TESTING ROUND {round_num}")
        print(f"{'=' * 80}\n")

        # Update model path for each round
        global model_path
        model_path = os.path.join(
            save_dir,
            f'STCNN_frame_NO_DAVIS{frame}_round{round_num}',
            f'STCNN_frame_NO_DAVIS{frame}_round{round_num}_best.pth'
        )

        if os.path.exists(model_path):
            main(frame=frame, epochs=epochs, test_thresholds=test_thresholds, model_round=round_num)
        else:
            print(f"Model not found: {model_path}")
            print("Skipping this round...")


if __name__ == "__main__":
    # Test with multiple thresholds
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Option 1: Test a single round
    main(frame=4, epochs=100, test_thresholds=test_thresholds, model_round=0)

    # Option 2: Test all rounds (uncomment to use)
    # test_all_rounds(frame=4, epochs=100, test_thresholds=test_thresholds)