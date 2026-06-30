from __future__ import division

import os
import numpy as np
import cv2
import torch
import re
import pickle
import random
import imageio
import cv2
import skimage.morphology as sm
from PIL import Image
from torch.utils.data import Dataset
from dataloaders import custom_transforms as tr

class FIREDatasetGeneral(Dataset):
	def __init__(self,
				inputRes=None,
				image_path="/home/r56x196/Data/archive-2/Image/Merged",
				mask_path="/home/r56x196/Data/archive-2/Mask/Fire",
				transform=None,
				num_frame=4):
		self.transform = transform
		self.inputRes = inputRes
		self.toTensor = tr.ToTensor()
		self.num_frame = num_frame

		self.split = image_path[-1]

		image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

		def numerical_key(filename):
			# Extract the first number found in the filename
			numbers = re.findall(r'\d+', filename)
			return int(numbers[0]) if numbers else -1

		self.image_files = sorted(
			[
				os.path.join(image_path, f)
				for f in os.listdir(image_path)
				if os.path.isfile(os.path.join(image_path, f))
				and os.path.splitext(f)[1].lower() in image_extensions
			],
			key=lambda x: numerical_key(os.path.basename(x))
		)

		self.masks = sorted(
			[
				os.path.join(mask_path, f)
				for f in os.listdir(mask_path)
				if os.path.isfile(os.path.join(mask_path, f))
				and os.path.splitext(f)[1].lower() in image_extensions
			],
			key=lambda x: numerical_key(os.path.basename(x))
		)
	def __len__(self):
		return len(self.image_files) - self.num_frame


	def __getitem__(self, idx):

		img = []
		for i in range(idx, idx + self.num_frame):
			imgs = imageio.imread(self.image_files[i])
			imgs = np.array(imgs, dtype=np.float32)
			if (self.inputRes is not None):
				imgs = cv2.resize(imgs, (self.inputRes[1], self.inputRes[0]))

			img.append(imgs)

		gt = cv2.imread(self.masks[idx + self.num_frame], 0)
		gt[gt == 1] = 255
		frame = imageio.imread(self.image_files[idx + self.num_frame])
		# Resize gt and frame
		if self.inputRes is not None:
			gt = cv2.resize(gt, (self.inputRes[1], self.inputRes[0]),
						interpolation=cv2.INTER_NEAREST)
			frame = cv2.resize(frame, (self.inputRes[1], self.inputRes[0]))

		print("split: ", self.split)
		img = np.concatenate(img,axis=2)

		img = np.array(img, dtype=np.float32)
		gt = np.array(gt, dtype=np.float32)
		frame = np.array(frame, dtype=np.float32)

		# normalize
		gt = gt / np.max([gt.max(), 1e-8])
		gt[gt > 0] = 1.0

		pred_gt = frame
		frame = frame / 255
		frame = np.subtract(frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
		frame = np.true_divide(frame,np.array([0.229, 0.224, 0.225], dtype=np.float32))

		sample = {'images': img, 'frame': frame, 'seg_gt': gt,'pred_gt': pred_gt}

		if self.transform is not None:
			sample = self.transform(sample)

		img = sample['images']
		img[np.isnan(img)] = 0.
		img[img > 255] = 255.0
		img[img < 0] = 0.
		img = img / 127.5 - 1.
		sample['images'] = img
		pred_gt = sample['pred_gt']
		pred_gt[np.isnan(pred_gt)] = 0.
		pred_gt[pred_gt > 255] = 255.0
		pred_gt[pred_gt < 0] = 0.
		pred_gt = pred_gt / 127.5 - 1.
		sample['pred_gt'] = pred_gt
		sample = self.toTensor(sample)
		return sample

class FIREDatasetSingle(Dataset):
	def __init__(self, inputRes=None,
			  	 samples_path="/home/c43n256/Data/Mask_Data",
				 transform=None,
				 mode="train"):
		self.transform = transform
		self.inputRes = inputRes
		self.toTensor = tr.ToTensor()

		image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

		def numerical_key(filename):
			# Extract the first number found in the filename
			numbers = re.findall(r'\d+', filename)
			return int(numbers[0]) if numbers else -1

		self.image_files = sorted(
			[
				os.path.join(samples_path, "Images",mode, f)
				for f in os.listdir(os.path.join(samples_path, "Images",mode))
				if os.path.isfile(os.path.join(samples_path, "Images",mode, f))
				and os.path.splitext(f)[1].lower() in image_extensions
			],
			key=lambda x: numerical_key(os.path.basename(x))
		)

		self.masks = sorted(
			[
				os.path.join(samples_path, "Masks",mode, f)
				for f in os.listdir(os.path.join(samples_path, "Masks",mode))
				if os.path.isfile(os.path.join(samples_path, "Masks",mode, f))
				and os.path.splitext(f)[1].lower() in image_extensions
			],
			key=lambda x: numerical_key(os.path.basename(x))
		)
	def __len__(self):
		return len(self.image_files)


	def __getitem__(self, idx):

		img = imageio.imread(self.image_files[idx])
		img = np.array(img, dtype=np.float32)
		if (self.inputRes is not None):
			img = cv2.resize(img, (self.inputRes[1], self.inputRes[0]))


		gt = cv2.imread(self.masks[idx], 0)
		gt[gt == 1] = 255
		frame = imageio.imread(self.image_files[idx])
		# Resize gt and frame
		if self.inputRes is not None:
			gt = cv2.resize(gt, (self.inputRes[1], self.inputRes[0]),
						interpolation=cv2.INTER_NEAREST)
			frame = cv2.resize(frame, (self.inputRes[1], self.inputRes[0]))

		img = np.array(img, dtype=np.float32)
		gt = np.array(gt, dtype=np.float32)
		frame = np.array(frame, dtype=np.float32)

		# normalize
		gt = gt / np.max([gt.max(), 1e-8])
		gt[gt > 0] = 1.0

		pred_gt = frame
		frame = frame / 255
		frame = np.subtract(frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
		frame = np.true_divide(frame,np.array([0.229, 0.224, 0.225], dtype=np.float32))

		sample = {'image': img, 'frame': frame, 'seg_gt': gt,'pred_gt': pred_gt}

		if self.transform is not None:
			sample = self.transform(sample)

		img = sample['image']
		img[np.isnan(img)] = 0.
		img[img > 255] = 255.0
		img[img < 0] = 0.
		img = img / 127.5 - 1.
		sample['image'] = img
		pred_gt = sample['pred_gt']
		pred_gt[np.isnan(pred_gt)] = 0.
		pred_gt[pred_gt > 255] = 255.0
		pred_gt[pred_gt < 0] = 0.
		pred_gt = pred_gt / 127.5 - 1.
		sample['pred_gt'] = pred_gt
		sample = self.toTensor(sample)
		return sample



import os, re
import numpy as np
import cv2
import imageio
import torch
from torch.utils.data import Dataset
from dataloaders import custom_transforms as tr  # assumes dict-aware ToTensor in your project


class FIREDataset(Dataset):
    def __init__(self, inputRes=None,
                 samples_path="/home/c43n256/Data/Mask_Data",
                 transform=None,
                 mode="test",
                 num_frame=4):
        self.transform = transform
        self.inputRes = inputRes
        self.num_frame = num_frame

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

        def numerical_key(filename):
            nums = re.findall(r'\d+', filename)
            return int(nums[0]) if nums else -1

        self.image_files = sorted(
            [
                os.path.join(samples_path, "Images", mode, f)
                for f in os.listdir(os.path.join(samples_path, "Images", mode))
                if os.path.isfile(os.path.join(samples_path, "Images", mode, f))
                   and os.path.splitext(f)[1].lower() in image_extensions
            ],
            key=lambda x: numerical_key(os.path.basename(x))
        )

        self.masks = sorted(
            [
                os.path.join(samples_path, "Masks", mode, f)
                for f in os.listdir(os.path.join(samples_path, "Masks", mode))
                if os.path.isfile(os.path.join(samples_path, "Masks", mode, f))
                   and os.path.splitext(f)[1].lower() in image_extensions
            ],
            key=lambda x: numerical_key(os.path.basename(x))
        )

    def __len__(self):
        return len(self.image_files) - self.num_frame

    def __getitem__(self, idx):
        # -----------------------
        # Load frames (np.float32)
        # -----------------------
        frames_np = []
        for i in range(idx, idx + self.num_frame):
            img_np = imageio.imread(self.image_files[i])
            img_np = np.asarray(img_np, dtype=np.float32)
            if self.inputRes is not None:
                img_np = cv2.resize(img_np, (self.inputRes[1], self.inputRes[0]))
            frames_np.append(img_np)

        # Concatenate along channels: H x W x (3*num_frame)
        img = np.concatenate(frames_np, axis=2).astype(np.float32)

        # -----------------------
        # Load mask and current frame
        # -----------------------
        gt = cv2.imread(self.masks[idx + self.num_frame], 0)
        frame = imageio.imread(self.image_files[idx + self.num_frame])

        if self.inputRes is not None:
            gt = cv2.resize(gt, (self.inputRes[1], self.inputRes[0]), interpolation=cv2.INTER_NEAREST)
            frame = cv2.resize(frame, (self.inputRes[1], self.inputRes[0]))

        gt = np.asarray(gt, dtype=np.float32)
        frame = np.asarray(frame, dtype=np.float32)

        # Normalize mask to {0,1}
        gt = gt / max(gt.max(), 1e-8)
        gt[gt > 0] = 1.0

        # Keep copy of frame for pred_gt
        pred_gt = frame.copy()

        # Normalize frame to ImageNet stats
        frame = frame / 255.0
        frame = frame - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        frame = frame / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # -----------------------
        # Build sample dict
        # -----------------------
        sample = {'images': img, 'frame': frame, 'seg_gt': gt, 'pred_gt': pred_gt}

        # -----------------------
        # Apply custom transforms (if any)
        # -----------------------
        if self.transform is not None:
            sample = self.transform(sample)

        # -----------------------
        # Clean and normalize images and pred_gt
        # -----------------------
        # Handle 'images'
        im = sample['images']
        im = np.nan_to_num(im, nan=0.0, posinf=255.0, neginf=0.0)
        im = np.clip(im, 0.0, 255.0)
        im = im / 127.5 - 1.0
        sample['images'] = im

        # Handle 'pred_gt'
        pg = sample['pred_gt']
        pg = np.nan_to_num(pg, nan=0.0, posinf=255.0, neginf=0.0)
        pg = np.clip(pg, 0.0, 255.0)
        pg = pg / 127.5 - 1.0
        sample['pred_gt'] = pg

        # -----------------------
        # Convert all to tensors
        # -----------------------
        # Convert 'images' (H, W, C*num_frame) -> (C*num_frame, H, W)
        if not torch.is_tensor(sample['images']):
            sample['images'] = torch.from_numpy(sample['images']).permute(2, 0, 1).float()

        # Convert 'frame' (H, W, 3) -> (3, H, W)
        if not torch.is_tensor(sample['frame']):
            if sample['frame'].ndim == 3:
                sample['frame'] = torch.from_numpy(sample['frame']).permute(2, 0, 1).float()
            else:
                sample['frame'] = torch.from_numpy(sample['frame']).unsqueeze(0).float()

        # Convert 'seg_gt' (H, W) -> (1, H, W)
        if not torch.is_tensor(sample['seg_gt']):
            if sample['seg_gt'].ndim == 2:
                sample['seg_gt'] = torch.from_numpy(sample['seg_gt']).unsqueeze(0).float()
            else:
                sample['seg_gt'] = torch.from_numpy(sample['seg_gt']).float()

        # Convert 'pred_gt' (H, W, 3) -> (3, H, W)
        if not torch.is_tensor(sample['pred_gt']):
            if sample['pred_gt'].ndim == 3:
                sample['pred_gt'] = torch.from_numpy(sample['pred_gt']).permute(2, 0, 1).float()
            else:
                sample['pred_gt'] = torch.from_numpy(sample['pred_gt']).unsqueeze(0).float()

        return sample



import os
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class FIREDatasetSegmentation(Dataset):
    def __init__(self,
                 inputRes=None,
                 samples_path="/home/c43n256/Data/Mask_Data",
                 joint_transform=None,
                 transform=None,
                 target_transform=None,
                 mode="train",  # "train" or "test"
                 test_size=0.2,  # Fraction of data to use for testing
                 random_state=42):  # For reproducible splits

        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.inputRes = inputRes
        self.mode = mode

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

        def numerical_key(filename):
            # Extract the first number found in the filename
            numbers = re.findall(r'\d+', filename)
            return int(numbers[0]) if numbers else -1

        # Set paths for combined folders
        image_path = os.path.join(samples_path, "Images/combined")
        mask_path = os.path.join(samples_path, "Masks/combined")

        # Check if the combined folders exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Images folder not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Masks folder not found: {mask_path}")

        # Get all image and mask files
        all_image_files = sorted(
            [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if os.path.isfile(os.path.join(image_path, f))
                   and os.path.splitext(f)[1].lower() in image_extensions
            ],
            key=lambda x: numerical_key(os.path.basename(x))
        )

        all_mask_files = sorted(
            [
                os.path.join(mask_path, f)
                for f in os.listdir(mask_path)
                if os.path.isfile(os.path.join(mask_path, f))
                   and os.path.splitext(f)[1].lower() in image_extensions
            ],
            key=lambda x: numerical_key(os.path.basename(x))
        )

        # Verify we have the same number of images and masks
        if len(all_image_files) != len(all_mask_files):
            raise ValueError(f"Mismatch: {len(all_image_files)} images vs {len(all_mask_files)} masks")

        if len(all_image_files) == 0:
            raise ValueError("No image files found in the specified directory")

        # Create indices for splitting
        all_indices = list(range(len(all_image_files)))

        # Split indices into train and test
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        # Select the appropriate indices based on mode
        if mode == "train":
            selected_indices = train_indices
        elif mode == "test":
            selected_indices = test_indices
        else:
            raise ValueError(f"Mode must be 'train' or 'test', got '{mode}'")

        # Create the final lists based on selected indices
        self.image_files = [all_image_files[i] for i in sorted(selected_indices)]
        self.masks = [all_mask_files[i] for i in sorted(selected_indices)]

        print(f"FIREDatasetSegmentation initialized for '{mode}' mode:")
        print(f"  Total images/masks: {len(all_image_files)}/{len(all_mask_files)}")
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Test samples: {len(test_indices)}")
        print(f"  Using {len(self.image_files)} samples for {mode}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Load images
            frame = Image.open(self.image_files[idx]).convert('RGB')
            gt = Image.open(self.masks[idx]).convert('L')

            # Ensure same size BEFORE converting to numpy
            # This step is important because we need to match sizes while still in PIL format
            if frame.size != gt.size:
                gt = gt.resize(frame.size, Image.NEAREST)

            # Apply resize if needed (still in PIL format)
            if self.inputRes is not None:
                frame = frame.resize((self.inputRes[1], self.inputRes[0]), Image.BILINEAR)
                gt = gt.resize((self.inputRes[1], self.inputRes[0]), Image.NEAREST)

            # Convert gt to numpy and normalize
            if hasattr(gt, 'numpy'):
                gt = gt.numpy()
            else:
                gt = np.array(gt)

            gt = gt.astype(np.float32)
            gt[gt > 0] = 1.0

            # NEW (Numpy arrays passed to transforms):
            # Convert PIL Images to numpy arrays BEFORE transforms
            frame = np.array(frame).astype(np.float32)  # Shape: (H, W, 3)
            gt = np.array(gt).astype(np.float32)  # Shape: (H, W)

            # Create sample dictionary with numpy arrays
            sample = {'images': frame, 'seg_gt': gt}

            if self.transform is not None:
                sample = self.transform(sample)

            return sample

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"Image file: {self.image_files[idx]}")
            print(f"Mask file: {self.masks[idx]}")
            raise


import os
import re
import cv2
import numpy as np
import imageio
from torch.utils.data import Dataset
import torchvision.transforms as tr
from sklearn.model_selection import train_test_split


class FIREDatasetRandom(Dataset):
    def __init__(self, inputRes=None,
                 samples_path="/home/c43n256/Data/Mask_Data",
                 transform=None,
                 mode="train",  # "train" or "test"
                 num_frame=4,
                 test_size=0.2,  # Fraction of data to use for testing
                 random_state=42):  # For reproducible splits

        self.transform = transform
        self.inputRes = inputRes
        self.toTensor = tr.ToTensor()
        self.num_frame = num_frame
        self.mode = mode

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

        def numerical_key(filename):
            # Extract the first number found in the filename
            numbers = re.findall(r'\d+', filename)
            return int(numbers[0]) if numbers else -1

        # Load all images and masks from single directories
        images_path = os.path.join(samples_path, "Images/combined")
        masks_path = os.path.join(samples_path, "Masks/combined")

        # Check if directories exist
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Images folder not found: {images_path}")
        if not os.path.exists(masks_path):
            raise FileNotFoundError(f"Masks folder not found: {masks_path}")

        # Get all image and mask files from single directories
        all_image_files = sorted(
            [
                os.path.join(images_path, f)
                for f in os.listdir(images_path)
                if os.path.isfile(os.path.join(images_path, f))
                   and os.path.splitext(f)[1].lower() in image_extensions
            ],
            key=lambda x: numerical_key(os.path.basename(x))
        )

        all_mask_files = sorted(
            [
                os.path.join(masks_path, f)
                for f in os.listdir(masks_path)
                if os.path.isfile(os.path.join(masks_path, f))
                   and os.path.splitext(f)[1].lower() in image_extensions
            ],
            key=lambda x: numerical_key(os.path.basename(x))
        )

        # Verify we have the same number of images and masks
        if len(all_image_files) != len(all_mask_files):
            print(f"Warning: Number of images ({len(all_image_files)}) != number of masks ({len(all_mask_files)})")

        # For sequence data, we need to account for num_frame requirement
        # We can only create sequences where we have enough consecutive frames
        max_valid_index = len(all_image_files) - num_frame
        if max_valid_index <= 0:
            raise ValueError(f"Not enough images for num_frame={num_frame}. Need at least {num_frame + 1} images.")

        # Create valid starting indices for sequences
        valid_indices = list(range(max_valid_index))

        # Split the valid indices into train and test
        train_indices, test_indices = train_test_split(
            valid_indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        # Select the appropriate indices based on mode
        if mode == "train":
            selected_indices = train_indices
        elif mode == "test":
            selected_indices = test_indices
        else:
            raise ValueError(f"Mode must be 'train' or 'test', got '{mode}'")

        # Store all files and the valid indices for this mode
        self.all_image_files = all_image_files
        self.all_mask_files = all_mask_files
        self.valid_indices = sorted(selected_indices)

        print(f"FIREDataset initialized for '{mode}' mode:")
        print(f"  Total images/masks: {len(all_image_files)}/{len(all_mask_files)}")
        print(f"  Valid sequence starting positions: {len(valid_indices)}")
        print(f"  Train sequences: {len(train_indices)}")
        print(f"  Test sequences: {len(test_indices)}")
        print(f"  Using {len(self.valid_indices)} sequences for {mode}")
        print(f"  Frames per sequence: {num_frame}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual starting index in the full dataset
        actual_start_idx = self.valid_indices[idx]

        img = []
        # Load sequence of frames
        for i in range(actual_start_idx, actual_start_idx + self.num_frame):
            imgs = imageio.imread(self.all_image_files[i])
            imgs = np.array(imgs, dtype=np.float32)
            if (self.inputRes is not None):
                imgs = cv2.resize(imgs, (self.inputRes[1], self.inputRes[0]))
            img.append(imgs)

        # Load ground truth mask and target frame
        target_frame_idx = actual_start_idx + self.num_frame
        gt = cv2.imread(self.all_mask_files[target_frame_idx], 0)
        gt[gt == 1] = 255
        frame = imageio.imread(self.all_image_files[target_frame_idx])

        # Resize gt and frame
        if self.inputRes is not None:
            gt = cv2.resize(gt, (self.inputRes[1], self.inputRes[0]),
                            interpolation=cv2.INTER_NEAREST)
            frame = cv2.resize(frame, (self.inputRes[1], self.inputRes[0]))

        # Concatenate sequence frames
        img = np.concatenate(img, axis=2)

        img = np.array(img, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)
        frame = np.array(frame, dtype=np.float32)

        # normalize gt
        gt = gt / np.max([gt.max(), 1e-8])
        gt[gt > 0] = 1.0

        pred_gt = frame
        # Normalize frame with ImageNet statistics
        frame = frame / 255
        frame = np.subtract(frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
        frame = np.true_divide(frame, np.array([0.229, 0.224, 0.225], dtype=np.float32))

        sample = {'images': img, 'frame': frame, 'seg_gt': gt, 'pred_gt': pred_gt}

        if self.transform is not None:
            sample = self.transform(sample)

        # Post-process images
        img = sample['images']
        img[np.isnan(img)] = 0.
        img[img > 255] = 255.0
        img[img < 0] = 0.
        img = img / 127.5 - 1.
        sample['images'] = img

        # Post-process pred_gt
        pred_gt = sample['pred_gt']
        pred_gt[np.isnan(pred_gt)] = 0.
        pred_gt[pred_gt > 255] = 255.0
        pred_gt[pred_gt < 0] = 0.
        pred_gt = pred_gt / 127.5 - 1.
        sample['pred_gt'] = pred_gt

        # Apply toTensor to individual arrays, not the entire sample dict
        sample['images'] = self.toTensor(sample['images'])
        sample['frame'] = self.toTensor(sample['frame'])
        sample['seg_gt'] = self.toTensor(sample['seg_gt'])
        sample['pred_gt'] = self.toTensor(sample['pred_gt'])
        return sample

