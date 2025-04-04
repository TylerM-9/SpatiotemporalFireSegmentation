from typing import List, Tuple, Dict, Optional
from network.joint_pred_seg import STCNN,FramePredDecoder,FramePredEncoder,SegEncoder,JointSegDecoder, SegBranch,SegEncoder
from network.googlenet import Inception3


class SegmentationBottleneck():

class SegmentationDataset():

class TemporalDataset():

import os
from typing import List, Optional, Callable, Tuple, Dict, Any
import numpy as np
import cv2
import imageio
from torch.utils.data import Dataset
import torchvision.transforms as tr


import os
from typing import List, Optional, Callable, Tuple, Dict, Any
import numpy as np
import cv2
import imageio
from torch.utils.data import Dataset
import torchvision.transforms as tr


class JointDataset(Dataset):
    def __init__(
        self,
        image_files: List[str],
        mask_files: List[str],
        inputRes: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        num_frame: int = 4
    ) -> None:
        """
        Base dataset for temporally ordered image-mask sequences.

        Args:
            image_files (List[str]): Sorted list of image file paths.
            mask_files (List[str]): Sorted list of mask file paths.
            inputRes (Optional[Tuple[int, int]]): Resize target resolution (H, W). Defaults to None.
            transform (Optional[Callable]): Transform function applied to each sample. Defaults to None.
            num_frame (int): Number of temporal frames used per sample. Defaults to 4.
        """
        assert len(image_files) == len(mask_files), "Mismatch between number of images and masks."

        self.image_files = image_files
        self.mask_files = mask_files
        self.inputRes = inputRes
        self.transform = transform
        self.num_frame = num_frame
        self.toTensor = tr.ToTensor()

    def __len__(self) -> int:
        return len(self.image_files) - self.num_frame

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Temporal frames collection
        frame_sequence: List[np.ndarray] = []

        for i in range(idx, idx + self.num_frame):
            frame: np.ndarray = imageio.imread(self.image_files[i]).astype(np.float32)
            if self.inputRes is not None:
                frame = cv2.resize(frame, (self.inputRes[1], self.inputRes[0]))
            frame_sequence.append(frame)

        # Load the last frame and corresponding GT mask
        last_frame: np.ndarray = imageio.imread(self.image_files[idx + self.num_frame]).astype(np.float32)
        temp_gt: np.ndarray = cv2.imread(self.mask_files[idx + self.num_frame], cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if self.inputRes is not None:
            last_frame = cv2.resize(last_frame, (self.inputRes[1], self.inputRes[0]))
            temp_gt = cv2.resize(temp_gt, (self.inputRes[1], self.inputRes[0]), interpolation=cv2.INTER_NEAREST)

        temp_gt[temp_gt == 1] = 255
        temp_gt /= max(temp_gt.max(), 1e-8)
        temp_gt[temp_gt > 0] = 1.0

        # Normalize last frame
        last_frame_normalized = last_frame / 255.0
        last_frame_normalized = (last_frame_normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Stack the frames along the channel axis
        frame = np.concatenate(frame_sequence, axis=2)
        frame = np.clip(frame, 0, 255) / 127.5 - 1.0

        # Normalize the predicted ground truth (temp_gt)
        temp_gt_normalized = np.clip(temp_gt, 0, 255) / 127.5 - 1.0

        sample: Dict[str, Any] = {
            'frames': frame,
            'last_frame': last_frame_normalized,
            'seg_gt': temp_gt,
            'temp_gt': last_frame
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return self.toTensor(sample)


class ModelConstractor:
    """Model Constractor class used for easy training and testing of STCNN model with different modules and datasets"""

    def __init__(
        self, 
        temporal: bool = True, 
        num_frames: Optional[int],
        seg_bottleneck: Optional[SegmentationBottleneck],
        seg_datasets: List[SegmentationDataset],
        temp_datasets: Optional[List[TemporalDataset]],
        joint_datasets: Optional[List[JointDataset]]
        ) -> None:
        """Initialize Model Constactor

        Args:
            temporal (bool, optional): Whether to use temporal branch. Defaults to True.
            seg_bottleneck (Optional[SegmentationBottleneck]): An optional segmentation bottleneck model.
            seg_datasets (List[SegmentationDataset]): List of segmentation datasets.
            temp_datasets (Optional[List[TemporalDataset]]): Optional list of temporal datasets.
            joint_datasets (Optional[List[JointDataset]]): Optional list of datasets that combine segmentation and temporal data.

        Returns:
            None
        """

        if temporal and num_frames is None:
            raise ValueError("`num_frames` must be provided when `temporal` is True")

        if temporal:
            self.seg_decoder = JointSegDecoder()
            self.temp_encoder = FramePredEncoder()
            self.temp_decoder = FramePredDecoder()
        else:
            self.seg_decoder = SegDecoder()

        self.seg_encoder = SegEncoder()

        self.net = STCNN(
            self.temp_encoder, 
            self.seg_encoder, 
            self.temp_decoder, 
            self.seg_decoder
        )

        self.joint_datasets = joint_datasets
    

    def _get_dataloaders(
        joint_datasets: Optional[List[JointDataset]],
        batch_size: int,
        num_workers: int = 4
        ) -> List[DataLoader]:
        """
        Returns a list of DataLoader instances for the given joint datasets.

        Args:
            joint_datasets (Optional[List[JointDataset]]): A list of JointDataset instances for which 
                DataLoaders need to be created. If no datasets are provided, an empty list is returned.
            batch_size (int): The batch size to be used for each DataLoader.
            num_workers (int, optional): The number of subprocesses to use for data loading. Default is 4.

        Returns:
            List[DataLoader]: A list of DataLoader instances, one for each dataset in `joint_datasets`.
        """
        dataloaders = []
        
        # If joint_datasets is not empty, create a DataLoader for each dataset
        if joint_datasets:
            for dataset in joint_datasets:
                dataloaders.append(
                    DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True  # This can be useful when using CUDA
                    )
                )
    
        return dataloaders


    def _train_joint(
        epochs: int,
        batch_size: int,
        model_name: str,
        snapshot: int = 10
        ) -> None:

        print("Training Jointly: ")

        dataloaders = self._get_dataloaders(
            joint_datasets=self.joint_datasets,
            batch_size=batch_size
        )

        real_label = torch.ones(batch_size).float().to(device)
	    fake_label = torch.zeros(batch_size).float().to(device)

        dataset_losses = {f"dataset_{i}": [] for i in range(len(dataloaders))}  # 

        for epoch in range(epochs):

            epoch_loss = {f"dataset_{i}": 0.0 for i in range(len(dataloaders))}

            for dataloader_idx, dataloader in enumerate(dataloaders):

                for ii, sample_batched in enumerate(dataloader):

                            sample: Dict[str, Any] = {
                    'frames': frame,
                    'last_frame': last_frame_normalized,
                    'seg_gt': temp_gt,
                    'temp_gt': temp_gt_normalized
                }

                    frames, last_frame, temp_gt, seg_gt = sample_batched["frames"], sample_batched["last_frame"], sample_batched["temp_gt"], sample_batched["seg_gt"]

                    frames.requires_grad_()
			        last_frame.requires_grad_()

                    frames, last_frame, temp_gt, seg_gt = frames.to(device), last_frame.to(device), temp_gt.to(device), seg_gt.to(device)

                    temp_gt = F.upsample(temp_gt, size=(100, 178), mode='bilinear', align_corners=False)

                    temp_gt = temp_gt.detach()
                    seg_pred, temp_pred = net.forward(frames, last_frame)

                    D_real = netD(temp_gt).squeeze(1)
                    errD_real = criterion(D_real, real_label)
                    D_fake = netD(temp_pred.detach()).squeeze(1)
                    errD_fake = criterion(D_fake, fake_label)

                    optimizer.zero_grad()
                    seg_loss = seg_criterion(seg_pred[-1], seg_gt)
                    for i in reversed(range(len(seg_pred) - 1)):
                        seg_loss = seg_loss + (1 - curr_iter / iter_num) * seg_criterion(seg_pred[i],seg_gt)

                    seg_loss.backward()
                    optimizer.step()
                    curr_iter += 1

                    epoch_loss += seg_loss.item() 
                    if updateD:
                        ############################
                        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                        ###########################
                        # train with real
                        netD.zero_grad()
                        # train with fake
                        d_loss = errD_fake + errD_real
                        d_loss.backward()
                        optimizerD.step()

                    if updateG:
                        ############################
                        # (2) Update G network: maximize log(D(G(z)))
                        ###########################
                        optimizerG.zero_grad()
                        D_fake = netD(pred).squeeze(1)
                        errG = criterion(D_fake, real_label)

                        lp_loss = lp_function(pred, pred_gts)
                        total_loss = lp_loss + beta * errG
                        total_loss.backward()
                        optimizerG.step()
                dataset_losses[f"dataset_{dataloader_idx}"].append(dataset_loss)

            print(f"Epoch [{epoch + 1}/{epochs}]")
            for dataloader_idx in range(len(dataloaders)):
                print(f"  Loss for dataset_{dataloader_idx}: {dataset_losses[f'dataset_{dataloader_idx}'][-1]}")

            if (epoch % snapshot) == snapshot - 1:
                torch.save(net.state_dict(), os.path.join(save_model_dir, modelName + str(epoch) + '.pth'))

        # After training is complete, plot the loss curves for each dataset
        plt.figure(figsize=(10, 5))

        # Plot the loss for each dataset
        for dataloader_idx in range(len(dataloaders)):
            plt.plot(range(1, epochs + 1), dataset_losses[f"dataset_{dataloader_idx}"], label=f"Dataset {dataloader_idx + 1}")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs for Each Dataset")
        plt.legend()
        plt.grid(True)

        # Save the figure instead of displaying it
        plt.savefig('training_loss_curve.png')  # Save as PNG (can also use .pdf, .jpg, etc.)
        plt.close()  # Close the plot after saving to free memory



