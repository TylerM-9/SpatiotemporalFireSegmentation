
from network.joint_pred_seg import STCNN,FramePredDecoder,FramePredEncoder,JointSegDecoder
from network.joint_pred_seg import SegBranch, SegDecoder,SegEncoder
import numpy as np
import os
from mypath import Path
import torch
import imageio
from dataloaders import FIRE_dataloader as db
from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils.data import DataLoader, ConcatDataset

def main(frame, epochs):
    gpu_id = 0
    num_frame = frame
    num_epochs = epochs
    modelName = 'STCNN_frame_'+str(num_frame)

    gpu_id = 0
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

    save_dir = Path.save_root_dir()
    save_model_dir = os.path.join(save_dir, modelName)

    seg_enc = SegEncoder()
    seg_dec = SegDecoder()

    net = SegBranch(net_enc=seg_enc,net_dec=seg_dec)
    print("Updationg weights from pretrained")
    net.load_state_dict(
        torch.load("/home/c43n256/STCNN/output/STCNN_frame_segmentation_only_full4/STCNN_frame_segmentation_only_full4Flame-100.pth",
                map_location=lambda storage, loc: storage))

    available_splits = 17
    general_test_set = []
    for i in range(1, available_splits+1):
        general_test_set.append(db.FIREDatasetGeneral(inputRes=(400,710), image_path=f"/home/c43n256/Data/archive-2/Image/split_{i}",
        mask_path=f"/home/c43n256/Data/archive-2/Mask/split_{i}",num_frame=num_frame))

    test_set = ConcatDataset(general_test_set)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)
    # test_set = db.FIREDataset(inputRes=(400,710),mode="test", num_frame=4)
    # test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)

    num_img_test = len(test_loader)

    net.to(device)

    iou = 0
    iou_mean = 0
    pa = 0
    dice = 0
    for ii, sample_batched in enumerate(test_loader):
        seqs, frame, gts, pred_gts = (
            sample_batched['images'].to(device), 
            sample_batched['frame'].to(device),
            sample_batched['seg_gt'].to(device),
            sample_batched['pred_gt'].to(device)
        ) 

        seg_res = net(frame)

        seg_pred = seg_res[-1][0, :, :, :].data.cpu().numpy()
        seg_pred = 1 / (1 + np.exp(-seg_pred))

        gt_sample = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])*255
        pred_gts_sample = pred_gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])*255

        iou += iou_score(gt_sample,seg_pred)
        pa += pixel_accuracy(gt_sample,seg_pred)
        dice += dice_coefficient(gt_sample,seg_pred)
        iou_mean += iou_score_mean(gt_sample,seg_pred)

        print("IoU ", ii, " :", iou_score(gt_sample,seg_pred))
        print("Pixel Accuracy ", ii, " :", pixel_accuracy(gt_sample,seg_pred))

        if ii % 20 == 1:

            seg_pred = seg_pred.transpose([1, 2, 0])*255
            frame_sample = pred_gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
            frame_sample = inverse_transform(frame_sample)*255
            gt_sample3 = np.concatenate([gt_sample,gt_sample,gt_sample],axis=2)

            seg_pred3 = np.concatenate([seg_pred,seg_pred,seg_pred],axis=2)
            samples1 = np.concatenate((seg_pred3,frame_sample), axis=0)
            imageio.imwrite(os.path.join("test_fire_general_%s_s.png" % ii), np.uint8(samples1))

    print("FINAL IoU: ", iou/num_img_test)
    print("FINAL Pixel Accuracy: ", pa/num_img_test)
    print("Final Dice:", dice/num_img_test)
    print("FINAL mean IoU: ", iou_mean/num_img_test)


def iou_score_mean(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)

    y_true = np.squeeze(y_true)
    y_pred_bin = np.squeeze(y_pred_bin)

    # Fix y_true to be {0,1}
    adj_y_true = y_true.copy()
    adj_y_true[adj_y_true == 255] = 1

    tp = np.sum((adj_y_true == 1) & (y_pred_bin == 1))
    tn = np.sum((adj_y_true == 0) & (y_pred_bin == 0))
    fp = np.sum((adj_y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((adj_y_true == 1) & (y_pred_bin == 0))

    iou_foreground = tp / (tp + fp + fn + 1e-8)
    iou_background = tn / (tn + fp + fn + 1e-8)
    
    mean_iou = 0.5 * (iou_foreground + iou_background)

    return mean_iou

def iou_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)

    y_true = np.squeeze(y_true)
    y_pred_bin = np.squeeze(y_pred_bin)

    # Fix y_true to be {0,1}
    adj_y_true = y_true.copy()
    adj_y_true[adj_y_true == 255] = 1

    tp = np.sum((adj_y_true == 1) & (y_pred_bin == 1))
    tn = np.sum((adj_y_true == 0) & (y_pred_bin == 0))
    fp = np.sum((adj_y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((adj_y_true == 1) & (y_pred_bin == 0))


    return tp / (tp + fn + fp)


def inverse_transform(images):
	return (images+1.)/2.

import numpy as np

def pixel_accuracy(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)

    y_true = np.squeeze(y_true)
    y_pred_bin = np.squeeze(y_pred_bin)

    # Fix y_true to be {0,1}
    adj_y_true = y_true.copy()
    adj_y_true[adj_y_true == 255] = 1

    tp = np.sum((adj_y_true == 1) & (y_pred_bin == 1))
    tn = np.sum((adj_y_true == 0) & (y_pred_bin == 0))
    fp = np.sum((adj_y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((adj_y_true == 1) & (y_pred_bin == 0))

    m_pixel_accuracy = 1/2 * (tn / (tn + fp) + tp / (tp + fn))
    return m_pixel_accuracy

def dice_coefficient(y_true, y_pred, threshold=0.5):
    """Computes the Dice Coefficient for segmentation."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)

    y_true = np.squeeze(y_true)
    y_pred_bin = np.squeeze(y_pred_bin) 

    adj_y_true = y_true.copy()
    adj_y_true[adj_y_true == 255] = 1

    intersection = np.logical_and(adj_y_true, y_pred_bin).sum()
    return (2. * intersection) / (adj_y_true.sum() + y_pred_bin.sum() + 1e-8)


main(1, 149)