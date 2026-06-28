import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import binary_erosion, binary_dilation

def calculate_segmentation_metrics(pred, target, threshold=0.5):
    """
    Calculate common segmentation metrics
    
    Args:
        pred: Predicted segmentation mask (can be probabilities [0,1])
        target: Ground truth segmentation mask (binary)
        threshold: Threshold to binarize predictions if they're probabilities
    
    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    

    # Ensure binary masks
    if pred.max() > 1.0 or target.max() > 1.0:
        print("Warning: Input masks should be in range [0,1]")
        pred = pred / max(1.0, pred.max())
        target = target / max(1.0, target.max())
    
    # Binarize predictions if they're probabilistic
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > threshold).astype(np.uint8)
    
    # Calculate basic counts
    true_positive = np.sum((pred_binary == 1) & (target_binary == 1))
    false_positive = np.sum((pred_binary == 1) & (target_binary == 0))
    true_negative = np.sum((pred_binary == 0) & (target_binary == 0))
    false_negative = np.sum((pred_binary == 0) & (target_binary == 1))
    
    # Edge case handling - avoid division by zero
    epsilon = 1e-7
    
    # Calculate metrics
    precision = true_positive / (true_positive + false_positive + epsilon) * 100
    recall = sensitivity = true_positive / (true_positive + false_negative + epsilon) * 100
    specificity = true_negative / (true_negative + false_positive + epsilon) * 100
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    
    # Calculate IOU (Intersection over Union) / Jaccard Index
    intersection = true_positive
    union = true_positive + false_positive + false_negative
    iou = intersection / (union + epsilon) * 100
    
    # Calculate AUC if predictions are probabilities
    # Flatten arrays for ROC calculation
    pred_flat = pred.flatten()
    target_flat = target_binary.flatten()
    
    try:
        fpr, tpr, _ = roc_curve(target_flat, pred_flat)
        roc_auc = auc(fpr, tpr) * 100
    except:
        roc_auc = 0
        print("Warning: Could not calculate AUC. Check input values.")
    
    return {
        "Precision(%)": precision.item(),
        "Recall(%)": recall.item(),
        "Sensitivity(%)": sensitivity.item(),
        "Specificity(%)": specificity.item(),
        "F1-Score(%)": f1_score.item(),
        "IOU(%)": iou.item(),
        "AUC(%)": roc_auc.item()
    }

def evaluate_batch_segmentation(predictions, targets, threshold=0.5):
    """
    Evaluate a batch of segmentation predictions
    
    Args:
        predictions: Batch of predicted masks [B, C, H, W] or [B, H, W]
        targets: Batch of target masks [B, C, H, W] or [B, H, W]
        threshold: Threshold for binarizing predictions
    
    Returns:
        dict: Dictionary of average metrics across the batch
    """
    batch_size = len(predictions)
    metrics_sum = {
        "Precision(%)": 0,
        "Recall(%)": 0,
        "Sensitivity(%)": 0,
        "Specificity(%)": 0,
        "F1-Score(%)": 0,
        "IOU(%)": 0,
        "AUC(%)": 0
    }
    
    for i in range(batch_size):
        # Handle multi-channel case (take first channel or average)
        pred = predictions[i]
        target = targets[i]
        
        if len(pred.shape) > 2 and pred.shape[0] > 1:
            # Multi-channel case (e.g., multiple classes)
            # For this example, we'll take the first channel, but you might
            # want to average across channels or evaluate each separately
            pred = pred[0]
            target = target[0]
        
        # Calculate metrics for this sample
        sample_metrics = calculate_segmentation_metrics(pred, target, threshold)
        
        # Add to running sum
        for key in metrics_sum:
            metrics_sum[key] += sample_metrics[key]
    
    # Calculate average
    metrics_avg = {key: value / batch_size for key, value in metrics_sum.items()}
    
    return metrics_avg

def display_metrics(metrics, decimal_places=2):
    """
    Format and display metrics nicely
    """
    print("\n--- Segmentation Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.{decimal_places}f}")
    print("----------------------------\n")
    
    return metrics

# Example usage:
if __name__ == "__main__":
    # Create some dummy data
    pred = np.random.rand(10, 1, 256, 256)  # Batch of 10 predictions
    target = (np.random.rand(10, 1, 256, 256) > 0.7).astype(np.float32)  # Batch of 10 targets
    
    # Evaluate
    metrics = evaluate_batch_segmentation(pred, target)
    display_metrics(metrics)