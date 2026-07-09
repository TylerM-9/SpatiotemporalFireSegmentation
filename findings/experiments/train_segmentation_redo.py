from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from datetime import datetime
import socket
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
import matplotlib.pyplot as plt
import torch.nn.functional as F

from network.joint_pred_seg import STCNN,FramePredDecoder,SegDecoderCBAM,FramePredEncoder,SegEncoder,SegDecoder, SegBranch
from network.googlenet import Inception3

from network.shuffle import PretrainedShuffleEncoder

from dataloaders import custom_transforms as tr
from dataloaders import DAVIS_dataloader as db
#from dataloaders import FIRE_dataloader as db
from mypath import Path

gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
def main(args):
	# # Select which GPU, -1 if CPU
	if torch.cuda.is_available():
		print(f"CUDA available, using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
	else:
		print("CUDA not available, using CPU.")

	# # Setting other parameters
	resume_epoch = 0 # Default is 0, change if want to resume
	nEpochs = 100 # Number of epochs for training (500.000/2079)
	batch_size = 1
	snapshot = 1  # Store a model every snapshot epochs
	pred_lr = 1e-8
	seg_lr = 1e-4
	lr_D = 1e-4
	wd = 5e-4
	beta = 0.001
	margin = 0.3

	updateD = True
	updateG = False
	num_frame =args.frame_nums

	modelName = 'STCNN_frame_segmentation_only_CBAM'+str(num_frame)

	save_dir = Path.save_root_dir()
	if not os.path.exists(save_dir):
		os.makedirs(os.path.join(save_dir))
	save_model_dir = os.path.join(save_dir, modelName)
	if not os.path.exists(save_model_dir):
		os.makedirs(os.path.join(save_model_dir))

	# Network definition

	seg_enc = SegEncoder()
	seg_dec = SegDecoderCBAM()
	if resume_epoch == 0:
		# Do not have pre-trained
		initialize_model(seg_enc, seg_dec, save_dir,num_frame=num_frame)
		net = SegBranch(net_enc=seg_enc,net_dec=seg_dec)
	else:
		net = SegBranch(net_enc=seg_enc,net_dec=seg_dec)
		print("Updationg weights from pretrained")
		net.load_state_dict(
			torch.load("/home/c43n256/STCNN/output/STCNN_frame_segmentation_only_noPPM4/STCNN_frame_segmentation_only_noPPM4Davis-99.pth",
					map_location=lambda storage, loc: storage))


	# Logging into Tensorboard
	log_dir = os.path.join(save_dir, 'JointPredSegNet_runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
	writer = SummaryWriter(log_dir=log_dir, comment='-parent')


	# PyTorch 0.4.0 style
	net.to(device)

	lp_function = nn.MSELoss().to(device)
	criterion = nn.BCELoss().to(device)
	seg_criterion = nn.BCEWithLogitsLoss().to(device)

	# Use the following optimizer
	optimizer = optim.SGD([
		{'params': [param for name, param in net.named_parameters()], 'lr': seg_lr},
	], weight_decay=wd, momentum=0.9)

	# Preparation of the data loaders
	# Define augmentation transformations as a composition
	composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
											tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
											])

	# Training dataset and its iterator

	#FIRE DATASET training
	#db_train = db.FIREDataset(inputRes=(400,710),transform=composed_transforms,mode="train", num_frame=num_frame)


	
	db_train = db.DAVISDataset(inputRes=(400,710),samples_list_file=os.path.join('/home/c43n256/STCNN/data/DAVIS16_samples_list.txt'),
							transform=composed_transforms,num_frame=num_frame)
	
	trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4)
		

	# test_set = db.FIREDataset(inputRes=(400,710),mode="test", num_frame=num_frame)
	# test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=True)

	num_img_tr = len(trainloader)
	iter_num = nEpochs * num_img_tr
	curr_iter = resume_epoch * num_img_tr
	print("Training Network")
	real_label = torch.ones(batch_size).float().to(device)
	fake_label = torch.zeros(batch_size).float().to(device)

	epoch_losses = []
	val_loss_list = []
	for epoch in range(resume_epoch, nEpochs):
		epoch_loss = 0
		num_batches = len(trainloader)
		start_time = timeit.default_timer()

		for ii, sample_batched in enumerate(trainloader):

			inputs, gts = sample_batched['frame'], sample_batched['seg_gt']
			inputs.requires_grad_()

			inputs, gts = inputs.to(device), gts.to(device)
			pred = net.forward(inputs)
			optimizer.zero_grad()

			seg_loss = seg_criterion(pred[-1], gts)
			for i in reversed(range(len(pred) - 1)):
				seg_loss = seg_loss + (1 - curr_iter / iter_num) * seg_criterion(pred[i],gts)
			# loss = criterion(pred, gts)
			seg_loss.backward()
			optimizer.step()
			curr_iter += 1

			epoch_loss += seg_loss.item() 
			if curr_iter % 5 == 0:
				print(
					"Iters: [%2d] time: %4.4f, loss: %.8f"
					% (curr_iter, timeit.default_timer() - start_time, seg_loss.item())
				)

		
		# avg_epoch_loss = epoch_loss / num_batches  # Compute average loss for the epoch
		# epoch_losses.append(avg_epoch_loss)  # Store epoch loss
		# print(f"Epoch [{epoch+1}/{nEpochs}] - Avg Loss: {avg_epoch_loss:.8f}")
		# val_loss = 0
		# for idx, sample in enumerate(test_loader):
		# 	seqs, frames, gts, pred_gts = sample['images'], sample['frame'],sample['seg_gt'], \
		# 								sample['pred_gt']

		# 	seqs, frames, gts, pred_gts = seqs.to(device), frames.to(device), gts.to(device),pred_gts.to(device)
		# 	seg_res = net.forward(frames)
			
		# 	seg_loss = seg_criterion(seg_res[-1], gts)

		# 	val_loss += seg_loss.item() 
		
		# num_samples = len(test_loader)
		# val_loss_list.append(val_loss/num_samples)
		

		if (epoch % snapshot) == snapshot - 1 and epoch != 0:
			torch.save(net.state_dict(), os.path.join(save_model_dir, modelName + 'Davis-' + str(epoch) + '.pth'))

	plt.figure(figsize=(8, 6))  # Set figure size (optional)
	plt.plot(range(resume_epoch, nEpochs), epoch_losses, marker='o', linestyle='-', label="Training Loss")
	# plt.plot(range(resume_epoch, nEpochs), val_loss_list, marker='s', linestyle='--', label="Validation Loss", color='r')
	plt.xlabel("Epochs")
	plt.ylabel("Average Loss")
	plt.title("Training & Validation No Segmentation CBAM")
	plt.legend()
	plt.grid(True)

	# Save the plot
	plt.savefig("Training & Validation No Segmentation CBAM.png", dpi=300, bbox_inches='tight')
	writer.close()

def inverse_transform(images):
	return (images+1.)/2.


def initialize_netD(netD,model_path):
	# Load the Inception-v3 model from torch hub with pretrained weights
	hub_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
	hub_model.eval()
	
	# Get the state dictionary from the hub model
	pretrained_dict = hub_model.state_dict()
	
	# Get the state dictionary of your netD
	model_dict = netD.state_dict()
	
	# Filter out unnecessary keys
	# Filter out fc layers to avoid size mismatch
	filtered_dict = {k: v for k, v in pretrained_dict.items() 
					if k in model_dict and not k.startswith('fc.')}
	
	# Update your netD's state dictionary with the pretrained weights
	model_dict.update(filtered_dict)
	netD.load_state_dict(model_dict)

def initialize_model( seg_enc, seg_dec,save_dir,num_frame=4):

	print("Loading weights from pretrained SegBranch")  
	pretrained_SegBranch_dict = torch.load("/home/c43n256/STCNN/output/Seg_Branch/Seg_Branch_epoch_fire_segmentation_only_noppm-11300.pth", map_location=torch.device(device))

	# Load encoder weights
	model_dict = seg_enc.state_dict()
	missing_keys_enc = []
	shape_mismatches_enc = []

	print(len(model_dict))

	# Now load the matching weights
	encoder_dict = {k[8:]: v for k, v in pretrained_SegBranch_dict.items() if k[:8] == "encoder."}
	print(len(encoder_dict))

	pretrained_dict = {}
	for k, v in encoder_dict.items():
		if k in model_dict:
			if v.shape == model_dict[k].shape:
				pretrained_dict[k] = v
			else: 
				shape_mismatches_enc.append(k)
		else:
			missing_keys_enc.append(k)
	print(len(pretrained_dict))
	model_dict.update(pretrained_dict)
	seg_enc.load_state_dict(model_dict)

	print("Encoder - Missing keys:", missing_keys_enc)
	print("Encoder - Shape mismatches:", shape_mismatches_enc)

	# Load decoder weights
	model_dict = seg_dec.state_dict()
	missing_keys_dec = []
	shape_mismatches_dec = []

	print(len(pretrained_SegBranch_dict))
	print(len(model_dict))
	# Now load the matching weights
	decoder_dict = {k[8:]: v for k, v in pretrained_SegBranch_dict.items() if k[:8] == "decoder."}
	print(len(decoder_dict))

	pretrained_dict = {}
	for k, v in decoder_dict.items():
		if k in model_dict:
			if v.shape == model_dict[k].shape:
				pretrained_dict[k] = v
			else: 
				shape_mismatches_enc.append(k)
		else:
			missing_keys_enc.append(k)
	print(len(pretrained_dict))
	model_dict.update(pretrained_dict)
	seg_dec.load_state_dict(model_dict)


	print("Decoder - Missing keys:", missing_keys_dec)
	print("Decoder - Shape mismatches:", shape_mismatches_dec)


if __name__ == "__main__":
	main_arg_parser = argparse.ArgumentParser(description="parser for train frame predict")

	main_arg_parser.add_argument("--frame_nums", type=int, default=4,
								help="input frame nums")

	args = main_arg_parser.parse_args()
	main(args)