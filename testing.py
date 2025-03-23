
from network.joint_pred_seg import STCNN,FramePredDecoder,FramePredEncoder,SegEncoder,JointSegDecoder
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
    num_frame = frame
    num_epochs = epochs
    modelName = 'STCNN_frame_'+str(num_frame)

    save_dir = Path.save_root_dir()
    save_model_dir = os.path.join(save_dir, modelName)

    seg_enc = SegEncoder()
    pred_enc = FramePredEncoder(frame_nums=num_frame)
    pred_dec = FramePredDecoder()
    j_seg_dec = JointSegDecoder()

    net = STCNN(pred_enc, seg_enc, pred_dec, j_seg_dec)
    print("Updating weights from: {}".format(
            os.path.join(save_model_dir, modelName + '_fire_epoch-' + str(num_epochs) + '.pth')))
    net.load_state_dict(
            torch.load(os.path.join(save_model_dir, modelName + '_fire_epoch-' + str(num_epochs) + '.pth'),
                        map_location=lambda storage, loc: storage))



    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
											  tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
											  ])

    datasets = []
    for i in range(1,18):

        db_test = db.FIREDatasetGeneral(inputRes=(256,256),image_path=f"/home/r56x196/Data/archive-2/Image/split_{i}", mask_path=f"/home/r56x196/Data/archive-2/Mask/split_{i}",transform=composed_transforms,num_frame=num_frame)
        datasets.append(db_test)

    combined_dataset = ConcatDataset(datasets)
    
    #db_test = db.FIREDataset(inputRes=(256,256),transform=composed_transforms,num_frame=num_frame)
    testloader = DataLoader(combined_dataset, batch_size=1, shuffle=True)
    num_img_test = len(testloader)

    iou = 0
    pa = 0
    for ii, sample_batched in enumerate(testloader):
        seqs, frames, gts, pred_gts = sample_batched['images'], sample_batched['frame'],sample_batched['seg_gt'], \
                                            sample_batched['pred_gt']

        seg_res, pred = net.forward(seqs, frames)

        seg_pred = seg_res[-1][0, :, :, :].data.cpu().numpy()
        seg_pred = 1 / (1 + np.exp(-seg_pred))

        gt_sample = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])*255
        pred_gts_sample = pred_gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])*255

        iou += iou_score(gt_sample,seg_pred)
        pa += pixel_accuracy(gt_sample,seg_pred)

        print("IoU ", ii, " :", iou_score(gt_sample,seg_pred))
        print("Pixel Accuracy ", ii, " :", pixel_accuracy(gt_sample,seg_pred))


        seg_pred = seg_pred.transpose([1, 2, 0])*255
        frame_sample = pred_gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
        frame_sample = inverse_transform(frame_sample)*255
        gt_sample3 = np.concatenate([gt_sample,gt_sample,gt_sample],axis=2)

        seg_pred3 = np.concatenate([seg_pred,seg_pred,seg_pred],axis=2)
        samples1 = np.concatenate((seg_pred3, gt_sample3, frame_sample), axis=0)


        if ii == 200:
            break
        imageio.imwrite(os.path.join("test_fire%s_s.png" % ii), np.uint8(samples1))

    print("FINAL IoU: ", iou/num_img_test)
    print("FINAL Pixel Accuracy: ", pa/num_img_test)

def iou_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)  # Thresholding prediction

    y_true = np.squeeze(y_true)
    y_pred_bin = np.squeeze(y_pred_bin) 

    intersection = np.logical_and(y_true, y_pred_bin).sum()
    union = np.logical_or(y_true, y_pred_bin).sum()

    return intersection / union if union != 0 else 0.0

def inverse_transform(images):
	return (images+1.)/2.

def pixel_accuracy(y_true, y_pred, threshold=0.5):


    y_pred_bin = (y_pred > threshold).astype(np.uint8)

    y_true = np.squeeze(y_true)
    y_pred_bin = np.squeeze(y_pred_bin) 

    correct_pixels = (y_true == y_pred_bin).sum()
    total_pixels = y_true.size

    return correct_pixels / total_pixels


main(4, 149)