
from network.joint_pred_seg import STCNN,FramePredDecoder,FramePredEncoder,JointSegDecoder
from network.joint_pred_seg import SegBranch, SegDecoder,SegEncoder
import numpy as np
import os
from mypath import Path
import torch
import imageio
from dataloaders.FIRE_dataloader import FIREDatasetSingle
from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils.data import DataLoader

def main(frame, epochs):
    gpu_id = 0
    num_frame = frame
    num_epochs = epochs
    modelName = 'STCNN_frame_'+str(num_frame)

    save_dir = Path.save_root_dir()
    save_model_dir = os.path.join(save_dir, modelName)

    seg_enc = SegEncoder()
    decoder = SegDecoder()

    net = SegBranch(net_enc=seg_enc,net_dec=decoder)

    print("Updating weights from: {}".format(
            os.path.join(save_model_dir, modelName + '_fire_epoch-' + str(num_epochs) + '.pth')))

    # Load the full model's weights
    full_model_weights = torch.load(os.path.join(save_model_dir, modelName + '_fire_epoch-' + str(num_epochs) + '.pth'),
                                    map_location=lambda storage, loc: storage)

    seg_model_weights = net.state_dict()

    # Adjust prefixes to match segmentation model
    adjusted_weights = {k.replace("seg_encoder", "encoder").replace("seg_decoder", "decoder"): v 
                        for k, v in full_model_weights.items()}

    # Now filter weights that exist in both models
    filtered_weights = {k: v for k, v in adjusted_weights.items() if k in seg_model_weights and seg_model_weights[k].shape == v.shape}

    # Load the filtered weights into the segmentation model
    missing_keys, unexpected_keys = net.load_state_dict(filtered_weights, strict=False)

    # Print missing and unexpected keys for debugging
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", adjusted_weights.keys())




    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                                tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
                                                ])
    db_test = FIREDatasetSingle(inputRes=(400,710),transform=composed_transforms,mode="test",num_frame=num_frame)

    testloader = DataLoader(db_test, batch_size=1, shuffle=True, num_workers=4)
    num_img_test = len(testloader)

    iou = 0
    pa = 0
    for ii, sample_batched in enumerate(testloader):
        seqs, frames, gts, pred_gts = sample_batched['image'], sample_batched['frame'],sample_batched['seg_gt'], \
                                            sample_batched['pred_gt']

        seg_res = net.forward(seqs)

        seg_pred = seg_res[-1][0, :, :, :].data.cpu().numpy()
        seg_pred = 1 / (1 + np.exp(-seg_pred))

        gt_sample = gts[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])*255

        iou += iou_score(gt_sample,seg_pred)
        pa += pixel_accuracy(gt_sample,seg_pred)

        print("IoU ", ii, " :", iou_score(gt_sample,seg_pred))
        print("Pixel Accuracy ", ii, " :", pixel_accuracy(gt_sample,seg_pred))


        seg_pred = seg_pred.transpose([1, 2, 0])*255
        frame_sample = frames[0, :, :, :].data.cpu().numpy().transpose([1, 2, 0])
        frame_sample = inverse_transform(frame_sample)*255
        gt_sample3 = np.concatenate([gt_sample,gt_sample,gt_sample],axis=2)

        seg_pred3 = np.concatenate([seg_pred,seg_pred,seg_pred],axis=2)
        samples1 = np.concatenate((seg_pred3, gt_sample3, frame_sample), axis=0)



        imageio.imwrite(os.path.join("test_fire_seg%s_s.png" % ii), np.uint8(samples1))

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