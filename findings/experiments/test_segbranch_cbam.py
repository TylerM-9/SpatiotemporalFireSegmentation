import os
import numpy as np
import torch
import imageio
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders.FIRE_dataloader import FIREDatasetSegmentation
from network.joint_pred_seg import SegBranch, SegDecoderCBAM, SegEncoder

def dice_coefficient(y_true, y_pred, threshold=0.5):
    """Computes the Dice Coefficient for segmentation."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    intersection = np.logical_and(y_true, y_pred_bin).sum()
    return (2. * intersection) / (y_true.sum() + y_pred_bin.sum() + 1e-8)

def iou_score(y_true, y_pred, threshold=0.5):
    """Computes the Intersection over Union (IoU) score."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    intersection = np.logical_and(y_true, y_pred_bin).sum()
    union = np.logical_or(y_true, y_pred_bin).sum()
    return intersection / union if union != 0 else 0.0

def pixel_accuracy(y_true, y_pred, threshold=0.5):
    """Computes the pixel-wise accuracy."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    return (y_true == y_pred_bin).sum() / y_true.size

def inverse_transform(images):
    return (images + 1.) / 2.

def load_model(model_path, device):
    """Loads the segmentation model."""
    encoder = SegEncoder()
    decoder = SegDecoderCBAM()
    net = SegBranch(net_enc=encoder, net_dec=decoder).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net

def evaluate_model(test_loader, model, device, save_results=True):
    """Evaluates the model on the test dataset."""
    total_iou, total_pa, total_dice = 0, 0, 0
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, sample in enumerate(test_loader):
        inputs, gts = sample['images'].to(device), sample['gts'].to(device)
        pred = model(inputs)
        pred_np = pred[-1].detach().cpu().numpy()
        gts_np = gts.detach().cpu().numpy()

        # Compute metrics
        total_iou += iou_score(gts_np, pred_np)
        total_pa += pixel_accuracy(gts_np, pred_np)
        total_dice += dice_coefficient(gts_np, pred_np)

        print(f"Iteration {idx} - IoU: {total_iou:.4f}, Pixel Acc: {total_pa:.4f}, Dice: {total_dice:.4f}")

        if save_results and idx % 20 == 1:
            save_visualization(inputs, gts_np, pred_np, save_dir, idx)
    
    num_samples = len(test_loader)
    print(f"FINAL IoU: {total_iou / num_samples:.4f}")
    print(f"FINAL Pixel Accuracy: {total_pa / num_samples:.4f}")
    print(f"FINAL Dice Coefficient: {total_dice / num_samples:.4f}")

def save_visualization(inputs, gts, pred, save_dir, idx):
    """Saves visualization of the predictions."""
    inputs_vis = inverse_transform(inputs[0].cpu().numpy().transpose([1, 2, 0])) * 255
    gt_vis = np.concatenate([gts[0].transpose(1, 2, 0)] * 3, axis=2) * 255
    pred_vis = np.concatenate([pred[0].transpose(1, 2, 0)] * 3, axis=2) * 255
    result_img = np.clip(np.concatenate((pred_vis, gt_vis, inputs_vis), axis=0), 0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(save_dir, f"test_fire_seg_{idx}.png"), result_img)

def main():
    """Main function to run the evaluation."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_set = FIREDatasetSegmentation(inputRes=(400, 710),
                                       image_path="/home/r56x196/Data/Mask_Data/Images/test",
                                       mask_path="/home/r56x196/Data/Mask_Data/Masks/test",
                                       transform=transforms.ToTensor(),
                                       target_transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=True)
    
    model_path = "/home/r56x196/STCNN/output/Seg_Branch_CBAM/Seg_Branch_CBAM_epoch_fire_segmentation_only_cbam-11300.pth"
    model = load_model(model_path, device)
    evaluate_model(test_loader, model, device)

if __name__ == "__main__":
    main()