from network.joint_pred_seg import STCNN, FramePredDecoder, FramePredEncoder, JointSegDecoder
from network.joint_pred_seg import SegBranch, SegDecoder, SegEncoder
import numpy as np
import os
from mypath import Path
import torch
import imageio
from dataloaders.FIRE_dataloader import FIREDatasetSegmentation
from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils.data import DataLoader

# Select which GPU, -1 if CPU
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

def dice_coefficient(y_true, y_pred, threshold=0.5):
    """
    Computes the Dice Coefficient for segmentation.
    """
    y_pred_bin = (y_pred > threshold).astype(np.uint8)

    y_true = np.squeeze(y_true)
    y_pred_bin = np.squeeze(y_pred_bin)

    intersection = np.logical_and(y_true, y_pred_bin).sum()
    dice = (2. * intersection) / (y_true.sum() + y_pred_bin.sum() + 1e-8)
    return dice

def iou_score(y_true, y_pred, threshold=0.5):
    """
    Computes the Intersection over Union (IoU) score.
    """
    y_pred_bin = (y_pred > threshold).astype(np.uint8)

    y_true = np.squeeze(y_true)
    y_pred_bin = np.squeeze(y_pred_bin)

    intersection = np.logical_and(y_true, y_pred_bin).sum()
    union = np.logical_or(y_true, y_pred_bin).sum()

    return intersection / union if union != 0 else 0.0

def pixel_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes the pixel-wise accuracy.
    """
    y_pred_bin = (y_pred > threshold).astype(np.uint8)

    y_true = np.squeeze(y_true)
    y_pred_bin = np.squeeze(y_pred_bin)

    correct_pixels = (y_true == y_pred_bin).sum()
    total_pixels = y_true.size

    return correct_pixels / total_pixels

def inverse_transform(images):
    return (images + 1.) / 2.

def main(epochs):

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()
    # Dataset
    test_set = FIREDatasetSegmentation(inputRes=(400, 710),
                                 image_path="/home/r56x196/Data/Mask_Data/Images/test",
                                 mask_path="/home/r56x196/Data/Mask_Data/Masks/test",
                                 transform=img_transform,
                                 target_transform=target_transform)

    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=True)

    # Model setup
    encoder = SegEncoder()

    decoder = SegDecoder()
    net = SegBranch(net_enc=encoder, net_dec=decoder)
    net.to(device)

    # Load pre-trained model
    model_path = f"/home/r56x196/STCNN/output/Seg_Branch/Seg_Branch_epoch_fire_segmentation_only -{epochs}.pth"
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    iou = 0
    pa = 0
    dice = 0

    # Inference loop
    for ii, sample_batched in enumerate(test_loader):
        inputs, gts = sample_batched['images'], sample_batched['gts']

        inputs, gts = inputs.to(device), gts.to(device)
        pred = net.forward(inputs)

        pred_np = pred[-1].detach().cpu().numpy()
        gts_np = gts.detach().cpu().numpy()

        iou += iou_score(gts_np, pred_np)
        pa += pixel_accuracy(gts_np, pred_np)
        dice += dice_coefficient(gts_np, pred_np)

        print(f"Iteration {ii} - IoU: {iou:.4f}, Pixel Accuracy: {pa:.4f}, Dice: {dice:.4f}")

        if ii % 20 == 1:
            inputs_vis = inputs[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
            inputs_vis = inverse_transform(inputs_vis)*255

            gt_vis = gts[0].cpu().numpy().transpose(1, 2, 0) * 255
            gt_vis = np.concatenate([gt_vis, gt_vis, gt_vis], axis=2)

            pred_vis = pred_np[0].transpose(1, 2, 0) * 255
            pred_vis = np.concatenate([pred_vis, pred_vis, pred_vis], axis=2)

            result_img = np.concatenate((pred_vis, gt_vis, inputs_vis), axis=0)
            result_img = np.clip(result_img, 0, 255).astype(np.uint8)

            save_dir = "."
            os.makedirs(save_dir, exist_ok=True)
            imageio.imwrite(os.path.join(save_dir, f"test_fire_seg_{ii}.png"), result_img)

    num_images = len(test_loader)
    print(f"FINAL IoU: {iou / num_images:.4f}")
    print(f"FINAL Pixel Accuracy: {pa / num_images:.4f}")
    print(f"FINAL Dice Coefficient: {dice / num_images:.4f}")

# Run the function
main(14999)
