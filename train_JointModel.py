from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ResT.models.rest import *
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

from network.joint_pred_seg import STCNN,FramePredDecoder,JointSegDecoderREST,FramePredEncoder,SegEncoder,JointSegDecoder, SegBranch
from network.googlenet import Inception3

from network.shuffle import PretrainedShuffleEncoder

from dataloaders import custom_transforms as tr
#from dataloaders import DAVIS_dataloader as db
from dataloaders import FIRE_dataloader as db
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
	resume_epoch = 1 # Default is 0, change if want to resume
	nEpochs = 201 # Number of epochs for training (500.000/2079)
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

	modelName = 'STCNN_frame_REST'+str(num_frame)

	save_dir = Path.save_root_dir()
	if not os.path.exists(save_dir):
		os.makedirs(os.path.join(save_dir))
	save_model_dir = os.path.join(save_dir, modelName)
	if not os.path.exists(save_model_dir):
		os.makedirs(os.path.join(save_model_dir))

	# Network definition

	netD = Inception3(num_classes=1, aux_logits=False, transform_input=True)
	# Do not have a pre-trained discriminator
	initialize_netD(netD,os.path.join('/home/r56x196/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4','NetD_epoch-99.pth'))
	seg_enc = rest_base()
	pred_enc = FramePredEncoder(frame_nums=num_frame)
	pred_dec = FramePredDecoder()
	j_seg_dec = JointSegDecoderREST()
	if resume_epoch == 0:
		# Do not have pre-trained
		initialize_model(pred_enc, seg_enc, pred_dec, j_seg_dec, save_dir,num_frame=num_frame)
		net = STCNN(pred_enc, seg_enc, pred_dec, j_seg_dec)
	else:
		net = STCNN(pred_enc, seg_enc, pred_dec, j_seg_dec)
		print("Updating weights from: {}".format(
			os.path.join(save_model_dir, modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
		net.load_state_dict(
			torch.load("/home/r56x196/STCNN/output/STCNN_frame_REST4/STCNN_frame_REST4Davis-99.pth",
					   map_location=lambda storage, loc: storage))


	# Logging into Tensorboard
	log_dir = os.path.join(save_dir, 'JointPredSegNet_runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
	writer = SummaryWriter(log_dir=log_dir, comment='-parent')


	# PyTorch 0.4.0 style
	net.to(device)
	netD.to(device)

	lp_function = nn.MSELoss().to(device)
	criterion = nn.BCELoss().to(device)
	seg_criterion = nn.BCEWithLogitsLoss().to(device)

	# Use the following optimizer
	optimizer = optim.SGD([
		{'params': [param for name, param in net.seg_encoder.named_parameters()], 'lr': seg_lr},
		{'params': [param for name, param in net.seg_decoder.named_parameters()], 'lr': seg_lr},
	], weight_decay=wd, momentum=0.9)

	optimizerG = optim.Adam([{'params': [param for name, param in net.pred_encoder.named_parameters()], 'lr': pred_lr},
		{'params': [param for name, param in net.pred_decoder.named_parameters()], 'lr': pred_lr},], lr=pred_lr, weight_decay=wd)

	optimizerD = optim.Adam(netD.parameters(), lr=lr_D, weight_decay=wd)
	# Preparation of the data loaders
	# Define augmentation transformations as a composition
	composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
											  tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)),
											  ])

	# Training dataset and its iterator

	# FIRE DATASET training
	
	"""
	db_train = db.DAVISDataset(inputRes=(224,224),samples_list_file=os.path.join('/home/r56x196/STCNN/data/DAVIS16_samples_list.txt'),
							   transform=composed_transforms,num_frame=num_frame)
	
	trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4)
	"""
	
	db_train = db.FIREDataset(inputRes=(224,224),transform=composed_transforms,mode="train", num_frame=num_frame)
	trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4)
	test_set = db.FIREDataset(inputRes=(224,224),mode="test", num_frame=num_frame)
	test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=True)
	
	


	num_img_tr = len(trainloader)
	iter_num = nEpochs * num_img_tr
	curr_iter = resume_epoch * num_img_tr
	print("Training Network")
	real_label = torch.ones(batch_size).float().to(device)
	fake_label = torch.zeros(batch_size).float().to(device)

	epoch_losses = []
	val_loss_list = []
	lp_loss = None 
	for epoch in range(resume_epoch, nEpochs):
		epoch_loss = 0
		num_batches = len(trainloader)
		start_time = timeit.default_timer()

		for ii, sample_batched in enumerate(trainloader):

			seqs, frames, gts, pred_gts = sample_batched['images'], sample_batched['frame'],sample_batched['seg_gt'], \
										 sample_batched['pred_gt']

			# Forward-Backward of the mini-batch
			seqs.requires_grad_()
			frames.requires_grad_()

			seqs, frames, gts, pred_gts = seqs.to(device), frames.to(device), gts.to(device),pred_gts.to(device)

			pred_gts = F.upsample(pred_gts, size=(100, 178), mode='bilinear', align_corners=False)

			pred_gts = pred_gts.detach()
			seg_res, pred = net.forward(seqs, frames)

			D_real_input = F.interpolate(pred_gts, size=(75, 75), mode='bilinear', align_corners=False)
			D_fake_input = F.interpolate(pred.detach(), size=(75, 75), mode='bilinear', align_corners=False)

			# Discriminator passes
			netD.eval()  # avoid batchnorm crash
			D_real = netD(D_real_input).squeeze(1)
			D_fake = netD(D_fake_input).squeeze(1)
			netD.train()

			# Compute discriminator losses
			errD_real = criterion(D_real, real_label)
			errD_fake = criterion(D_fake, fake_label)

			optimizer.zero_grad()
			seg_loss = seg_criterion(seg_res[-1], gts)
			for i in reversed(range(len(seg_res) - 1)):
				seg_loss = seg_loss + (1 - curr_iter / iter_num) * seg_criterion(seg_res[i],gts)

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
				optimizerG.zero_grad()

				# Use upsampled fake for generator training too
				netD.eval()
				D_fake = netD(D_fake_input).squeeze(1)
				netD.train()
				errG = criterion(D_fake, real_label)

				if pred.shape[-2:] != pred_gts.shape[-2:]:
					pred = F.interpolate(pred, size=pred_gts.shape[-2:], mode='bilinear', align_corners=False)

				lp_loss = lp_function(pred, pred_gts)

				total_loss = lp_loss + beta * errG
				total_loss.backward()
				optimizerG.step()

			if (errD_fake.data < margin).all() or (errD_real.data < margin).all():
				updateD = False
			if (errD_fake.data > (1. - margin)).all() or (errD_real.data > (1. - margin)).all():
				updateG = False
			if not updateD and not updateG:
				updateD = True
				updateG = True

			if (ii + num_img_tr * epoch) % 5 == 4 and lp_loss:
				print(
					"Iters: [%2d] time: %4.4f, lp_loss: %.8f, G_loss: %.8f,seg_loss: %.8f"
					% (ii + num_img_tr * epoch, timeit.default_timer() - start_time, lp_loss.item(),errG.item(), seg_loss.item())
				)
				print('updateD:', updateD, 'updateG:', updateG)

		
		avg_epoch_loss = epoch_loss / num_batches  # Compute average loss for the epoch
		epoch_losses.append(avg_epoch_loss)  # Store epoch loss
		print(f"Epoch [{epoch+1}/{nEpochs}] - Avg Loss: {avg_epoch_loss:.8f}")
		val_loss = 0
		for idx, sample in enumerate(test_loader):
			seqs, frames, gts, pred_gts = sample['images'], sample['frame'],sample['seg_gt'], \
										 sample['pred_gt']

			seqs, frames, gts, pred_gts = seqs.to(device), frames.to(device), gts.to(device),pred_gts.to(device)
			seg_res, pred = net.forward(seqs, frames)
			
			seg_loss = seg_criterion(seg_res[-1], gts)

			val_loss += seg_loss.item() 
		
		num_samples = len(test_loader)
		val_loss_list.append(val_loss/num_samples)
		
		
		
		
		if (epoch % snapshot) == snapshot - 1 and epoch != 0:
			torch.save(net.state_dict(), os.path.join(save_model_dir, modelName + 'Davis-' + str(epoch) + '.pth'))
	
	plt.figure(figsize=(8, 6))  # Set figure size (optional)
	plt.plot(range(resume_epoch, nEpochs), epoch_losses, marker='o', linestyle='-', label="Training Loss")
	plt.plot(range(resume_epoch, nEpochs), val_loss_list, marker='s', linestyle='--', label="Validation Loss", color='r')
	plt.xlabel("Epochs")
	plt.ylabel("Average Loss")
	plt.title("Training Rest Flame")
	plt.legend()
	plt.grid(True)

	# Save the plot
	plt.savefig("Training Rest Flame.png", dpi=300, bbox_inches='tight')
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

def initialize_model(pred_enc, seg_enc, pred_dec, j_seg_dec,save_dir,num_frame=4):
	print("Loading weights from pretrained NetG")
	pretrained_netG_dict = torch.load(os.path.join('/home/r56x196/ondemand/data/sys/myjobs/projects/default/4/output/FramePredModels/frame_nums_4', 'NetG_epoch-99.pth'), map_location=torch.device(device))

	model_dict = pred_enc.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	pred_enc.load_state_dict(model_dict)

	model_dict = pred_dec.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	pred_dec.load_state_dict(model_dict)


	print("Loading weights from pretrained SegBranch")  
	pretrained_SegBranch_dict = torch.load("/home/r56x196/STCNN/output/Seg_Branch_ShuffleNet/Seg_Branch_ShuffleNetRest-Epochs-19999.pth", map_location=torch.device(device))

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
	model_dict = j_seg_dec.state_dict()
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
	j_seg_dec.load_state_dict(model_dict)


	print("Decoder - Missing keys:", missing_keys_dec)
	print("Decoder - Shape mismatches:", shape_mismatches_dec)


if __name__ == "__main__":
	main_arg_parser = argparse.ArgumentParser(description="parser for train frame predict")

	main_arg_parser.add_argument("--frame_nums", type=int, default=4,
								 help="input frame nums")

	args = main_arg_parser.parse_args()
	main(args)