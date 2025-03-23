import os
import numpy as np
from pyLucid.patchPaint import paint
import cv2
from PIL import Image
from pyLucid.lucidDream import dreamData
from mypath import Path

db_root_dir = Path.db_root_dir()
with open(os.path.join(db_root_dir, 'ImageSets/480p/', 'val.txt')) as f:
	seqnames = f.readlines()
for i in range(len(seqnames)):
	seq_name = seqnames[i].strip().split(' ')[0]
	mask = seqnames[i].strip().split(' ')[1]
	print("Mask",mask)
	dream_dir = os.path.join("/home/r56x196/Data",'dream')
	if not os.path.exists(dream_dir):
		os.makedirs(os.path.join(dream_dir))
	print(db_root_dir)
	print(os.path.join(db_root_dir, seq_name.lstrip('/')))
	# Instead of using seq_name directly, get its directory:
	seq_dir = os.path.join(db_root_dir, os.path.dirname(seq_name.lstrip('/')))
	mask_dir = os.path.join(db_root_dir, os.path.dirname(mask.lstrip('/')))

	# Now list the directory contents
	names_img = np.sort(os.listdir(seq_dir))

	# Build the image list using the directory
	img_list = [os.path.join(seq_dir, x) for x in names_img]
	print("mask dir", mask_dir)
	name_label = np.sort(os.listdir(mask_dir))
	labels = [os.path.join(mask_dir, x) for x in name_label]
	img_path = img_list[0]
	label_path = labels[0]
	Iorg = cv2.imread(img_path)
	print(label_path)
	Morg = Image.open(label_path)
	palette = Morg.getpalette()
	Morg = np.array(Morg)
	Morg[Morg>0]=1

	print(Iorg.shape, np.array(Morg).shape)
	bg = paint(Iorg, np.array(Morg), False)
	cv2.imwrite(os.path.join("/home/r56x196/Data/dream",'bg.jpg'), bg)
	# bg = cv2.imread(os.path.join(save_path,'bg.jpg'))
	for i in range(100):
		im_1, gt_1, bb_1 = dreamData(Iorg, np.array(Morg), bg, False)
		# Image 1 in this pair.
		cv2.imwrite(os.path.join(dream_dir,'%03d.jpg'%i), im_1)

		# Mask for image 1.
		gtim1 = Image.fromarray(gt_1, 'P')
		gtim1.putpalette(palette)
		gtim1.save(os.path.join(dream_dir,'%03d.png'%i))
