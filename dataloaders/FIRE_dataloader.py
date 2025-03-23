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
				mask_path="/home/r56x196/Data/archive-2/Merged/Fire",
				transform=None,
				num_frame=4):
		self.transform = transform
		self.inputRes = inputRes
		self.toTensor = tr.ToTensor()
		self.num_frame = num_frame

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
			  	 samples_path="/home/r56x196/Data/Mask_Data",
				 transform=None,
				 mode="train"):
		self.transform = transform
		self.inputRes = inputRes
		self.toTensor = tr.ToTensor()
		self.num_frame = num_frame

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

	

class FIREDataset(Dataset):
	def __init__(self, inputRes=None,
			  	 samples_path="/home/r56x196/Data/Mask_Data",
				 transform=None,
				 mode="test",
				 num_frame=4):
		self.transform = transform
		self.inputRes = inputRes
		self.toTensor = tr.ToTensor()
		self.num_frame = num_frame

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

class FIREDatasetSegmentation(Dataset):
	def __init__(self,
					inputRes=None,
					image_path="/home/r56x196/Data/Mask_Data/Images/train",
					mask_path="/home/r56x196/Data/Mask_Data/Masks/train",
					joint_transform=None, 
					transform=None, 
					target_transform=None):
		self.joint_transform = joint_transform
		self.transform = transform
		self.target_transform = target_transform
		self.inputRes = inputRes
		image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

		def numerical_key(filename):
			# Extract the first number found in the filename
			numbers = re.findall(r'\d+', filename)
			return int(numbers[0]) if numbers else -1

		# Get sorted list of image and mask files
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
		return len(self.image_files) 

	def __getitem__(self, idx):
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

		# Apply transformations
		if self.joint_transform is not None:
			frame, gt = self.joint_transform(frame, gt)
		if self.transform is not None:
			frame = self.transform(frame)
		if self.target_transform is not None:
			gt = self.target_transform(gt)
		gt = gt.numpy()
		gt[gt > 0] = 1.0

		# Create sample dictionary
		sample = {'images': frame, 'gts': gt}

		return sample
