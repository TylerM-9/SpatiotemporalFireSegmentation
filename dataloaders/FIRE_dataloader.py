from __future__ import division

import os
import numpy as np
import cv2
import torch
import pickle
import random
import imageio
import cv2
from torch.utils.data import Dataset
from dataloaders import custom_transforms as tr

class DAVISDataset(Dataset):
	"""DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

	def __init__(self, inputRes=None,
				 samples_list_file='/home/xk/PycharmProjects/Pred_Seg/data/DAVIS16_samples_list.txt',
				 transform=None,
				 num_frame=4):

		f = open(samples_list_file, "r")
		lines = f.readlines()
		self.samples_list = lines
		self.transform = transform
		self.inputRes = inputRes
		self.toTensor = tr.ToTensor()
		self.num_frame = num_frame

	def __len__(self):
		return len(self.samples_list)

	def __getitem__(self, idx):

		sample_line = self.samples_list[idx].strip()
		seq_path_line,frame_path,gt_path = sample_line.split(';')
		seq_path = seq_path_line.split(',')
		imgs = []
		for i in range(self.num_frame):
			img = imageio.imread(seq_path[i])
			img = np.array(img, dtype=np.float32)
			if (self.inputRes is not None):
				img = cv2.resize(img, (self.inputRes[1], self.inputRes[0]))

			imgs.append(img)

		gt = cv2.imread(gt_path, 0)
		frame = imageio.imread(frame_path)

    # Resize gt and frame
		if self.inputRes is not None:
			gt = cv2.resize(gt, (self.inputRes[1], self.inputRes[0]), 
						interpolation=cv2.INTER_NEAREST)
			frame = cv2.resize(frame, (self.inputRes[1], self.inputRes[0]))
		imgs = np.concatenate(imgs,axis=2)

		imgs = np.array(imgs, dtype=np.float32)
		gt = np.array(gt, dtype=np.float32)
		frame = np.array(frame, dtype=np.float32)

		# normalize
		gt = gt / np.max([gt.max(), 1e-8])
		gt[gt > 0] = 1.0

		pred_gt = frame
		frame = frame / 255
		frame = np.subtract(frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
		frame = np.true_divide(frame,np.array([0.229, 0.224, 0.225], dtype=np.float32))

		sample = {'images': imgs, 'frame': frame, 'seg_gt': gt,'pred_gt': pred_gt}

		if self.transform is not None:
			sample = self.transform(sample)

		imgs = sample['images']
		imgs[np.isnan(imgs)] = 0.
		imgs[imgs > 255] = 255.0
		imgs[imgs < 0] = 0.
		imgs = imgs / 127.5 - 1.
		sample['images'] = imgs
		pred_gt = sample['pred_gt']
		pred_gt[np.isnan(pred_gt)] = 0.
		pred_gt[pred_gt > 255] = 255.0
		pred_gt[pred_gt < 0] = 0.
		pred_gt = pred_gt / 127.5 - 1.
		sample['pred_gt'] = pred_gt
		sample = self.toTensor(sample)
		return sample



class DAVIS_First_Frame_Dataset(Dataset):
	def __init__(self, train=True,
				 inputRes=None,
				 db_root_dir='/home/xk/Dataset/DAVIS/',
				 transform=None,
				 seq_name=None,
				 frame_nums=4):
		"""Loads image to label pairs for tool pose estimation
		db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
		"""
		self.train = train
		self.inputRes = inputRes
		self.db_root_dir = db_root_dir
		self.transform = transform
		self.seq_name = seq_name
		self.toTensor = tr.ToTensor()
		self.frame_nums = frame_nums
		# Initialize the per sequence images for online training

		names_img = np.sort([f for f in os.listdir(os.path.join(db_root_dir, 'first_frame/', str(seq_name),'dream'))
							 if f.endswith(".jpg")])
		img_list = list(map(lambda x: os.path.join(db_root_dir,'first_frame/', str(seq_name), 'dream', x), names_img))
		name_label = np.sort([f for f in os.listdir(os.path.join(db_root_dir, 'first_frame/', str(seq_name),'dream'))
							 if f.endswith(".png")])
		labels = list(map(lambda x: os.path.join(db_root_dir,'first_frame/', str(seq_name), 'dream', x), name_label))


		assert (len(labels) == len(img_list))

		self.img_list = img_list[:100]
		self.labels = labels[:100]

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, idx):
		imgs,frame, gt, pred_gt = self.make_img_gt_pair(idx)

		sample = {'images': imgs, 'frame': frame, 'seg_gt': gt, 'pred_gt': pred_gt}

		if self.seq_name is not None:
			fname = os.path.join(self.seq_name, "%05d" % idx)
			sample['fname'] = fname

		if self.transform is not None:
			sample = self.transform(sample)

		imgs = sample['images']
		imgs[np.isnan(imgs)] = 0.
		imgs[imgs > 255] = 255.0
		imgs[imgs < 0] = 0.
		imgs = imgs / 127.5 - 1.
		sample['images'] = imgs
		pred_gt = sample['pred_gt']
		pred_gt[np.isnan(pred_gt)] = 0.
		pred_gt[pred_gt > 255] = 255.0
		pred_gt[pred_gt < 0] = 0.
		pred_gt = pred_gt / 127.5 - 1.
		sample['pred_gt'] = pred_gt
		sample = self.toTensor(sample)

		return sample

	def make_img_gt_pair(self, idx):
		"""
		Make the image-ground-truth pair
		"""

		img = imageio.imread(self.img_list[idx])
		imgs = [img]
		for i in range(self.frame_nums-1):
			imgs.append(img)

		gt = cv2.imread(self.labels[idx], 0)
		frame = img
		imgs = np.concatenate(imgs,axis=2)

		if self.inputRes is not None:

			imgs = cv2.resize(imgs, (self.inputRes[1],self.inputRes[0]))
			gt = cv2.resize(gt, (self.inputRes[1],self.inputRes[0]),interpolation=cv2.INTER_NEAREST)
			frame = cv2.resize(frame, (self.inputRes[1],self.inputRes[0]))

		imgs = np.array(imgs, dtype=np.float32)
		gt = np.array(gt, dtype=np.float32)
		frame = np.array(frame, dtype=np.float32)
		# normalize

		# pred_gt = frame / 127.5 - 1.
		pred_gt = frame
		frame = frame / 255
		frame = np.subtract(frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
		frame = np.true_divide(frame, np.array([0.229, 0.224, 0.225], dtype=np.float32))
		gt = gt / np.max([gt.max(), 1e-8])
		gt[gt > 0] = 1.0

		return imgs, frame, gt, pred_gt
	




# Read the mask as a grayscale image
mask = cv2.imread('/Users/bezbodima/Projects/attentionCNN/STCNN/STCNN/data/Mask_Data/Masks/image_0.png', cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread('/Users/bezbodima/Downloads/DAVIS/Annotations/480p/bear/00001.png', cv2.IMREAD_GRAYSCALE)

unique_values = np.unique(mask)
print("Unique pixel values in the original mask:", unique_values)
unique_values = np.unique(mask2)
print("Unique pixel values in the original mask:", unique_values)
# Apply a binary threshold: pixels > 0 become 255, else 0.
mask[mask == 1] = 255
unique_values = np.unique(mask)
print("Unique pixel values in the original mask:", unique_values)
# Save or display the result
cv2.imwrite('binary_mask.png', mask)