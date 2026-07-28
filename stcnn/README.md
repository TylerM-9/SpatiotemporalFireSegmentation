# ST-UNet3+: Dual-Branch Spatiotemporal based on UNet3+ for Fire Segmentation

A spatiotemporal segmentation network that combines a **UNet3+ encoder/decoder** with a **temporal prediction branch** to detect and segment fire in video sequences.

The key idea: a pretrained temporal branch processes previous frames to predict motion context, and an attention module (`SimpleContextAdd`) fuses this temporal signal into the spatial segmentation decoder at multiple scales.

---

## Architecture Overview

<img width="424" height="831" alt="workflow_pipeline_newerer" src="https://github.com/user-attachments/assets/97032f84-8086-4f9e-b44b-1b9f6bc6a7ef" />

```

ST-UNET3+ ARCHITECTURE GOES HERE

```

The `SimpleContextAdd` attention module (in `network/UNET_ST.py`) integrates:
1. Current decoder features
2. Previous stage features (recurrence)
3. Temporal branch features (motion context)
4. High-level context features

---

## Repository Structure

```
.
├── train.py                   # Training script (DAVIS pre-training or FIRE fine-tuning)
├── test.py                    # Evaluation script (FIRE dataset, multiple thresholds)
├── mypath.py                  # Dataset and output path configuration
├── metrics.py                 # Segmentation evaluation metrics
│
├── network/
│   ├── UNET_ST.py             # Baseline model: UNetEncoder, UNetDecoder, STUNet, SimpleContextAdd
|   ├── STUnet3plus.py         # Proposed model: UNet3PlusEncoder, UNet3PlusDecoder, ST-UNet3+, SimpleContextAdd
│   └── joint_pred_seg.py      # Temporal branch: FramePredEncoder, FramePredDecoder
│
│
│
├── dataloaders/
│   ├── FIRE_dataloader.py     # FIRE dataset loader (FIREDatasetRandom, FIREDataset, ...)
│   ├── DAVIS_dataloader.py    # DAVIS-2016 dataset loader
│   └── custom_transforms.py  # Data augmentation transforms
│
├── layers/
│   └── layers.py              # Custom layer utilities (interp_surgery, DenseCRF)
│
├── data/
│   ├── DAVIS16_samples_list.txt
│   ├── DAVIS_seqs_list.txt
│   └── VID_seqs_list.txt
│
└── requirements.txt
```

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/TylerM-9/SpatiotemporalFireSegmentation.git
cd SpatiotemporalFireSegmentation/stcnn
pip install -r requirements.txt
```

> All commands below should be run from inside the `stcnn/` directory.

**2. Configure paths in `mypath.py`**

Edit `mypath.py` to point to your local dataset paths:

```python
class Path(object):
    @staticmethod
    def db_root_dir():
        return '/path/to/DAVIS'          # DAVIS-2016 root

    @staticmethod
    def save_root_dir():
        return '/path/to/output'         # Where checkpoints are saved
```

**3. Dataset structure**

FIRE dataset should be organized as:
```
/path/to/Mask_Data/
    Images/
        combined/        # All images in one flat directory
            00001.jpg
            00002.jpg
            ...
    Masks/
        combined/        # Corresponding binary masks
            00001.png
            00002.png
            ...
```

DAVIS-2016:
```
/path/to/DAVIS/
    JPEGImages/480p/<sequence>/
    Annotations/480p/<sequence>/
```

---

## Pretrained Weights Required

The temporal branch (`FramePredEncoder` / `FramePredDecoder`) must be initialized from pretrained frame-prediction weights before training the full ST-UNet and ST-UNet3+.

Update the paths in `train.py` (lines 91–103):
```python
pretrained_netG_dict = torch.load('/path/to/NetG_epoch-99.pth', ...)
initialize_netD(netD, '/path/to/NetD_epoch-99.pth')
```

---

## Training

**Stage 1 — Pre-train on DAVIS** (optional, helps initialization):
```bash
python train.py --dataset davis --frame_nums 4
```

**Stage 2 — Fine-tune on FIRE**:
```bash
python train.py --dataset fire --frame_nums 4
```

**Resume from checkpoint**:
```bash
python train.py --dataset fire --frame_nums 4 --resume_epoch 100
```

**Load pretrained segmentation weights**:
```bash
python train.py --dataset fire --frame_nums 4 --pretrained_seg /path/to/checkpoint.pth
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `201` | Number of epochs to train |
| `--model` | `stunet3plus` | Architecture choice: `stunet` or `stunet3plus` |
| `--frame_nums` | `4` | Number of temporal context frames |
| `--dataset` | `fire` | `fire` (train+val) or `davis` (train only) |
| `--resume_epoch` | `0` | Epoch to resume from (0 = fresh start) |
| `--pretrained_seg` | `None` | Path to pretrained segmentation checkpoint |
| `--output_dir` | `/home/.../output` | Directory to save models and logs |
| `--seed` | `42` | Random seed for reproducibility |
| `--lr` | `1e-4` | Learning rate for segmentation |
| `--wd` | `5e-4` | Weight decay penalty |
| `--batch` | `6` | Batch size for training |

Checkpoints are saved every 5 epochs to `{save_root_dir}/{model_name}/`.

---

## Evaluation

```bash
python test.py
```

Update the two variables at the top of `test.py`:
```python
model_path = "/path/to/STUNET_UNET_DAVIS_FIRE4-94.pth"
model_name  = "STUNET_UNET_FIRE4"
```

The script evaluates over multiple thresholds `[0.1, 0.2, ..., 0.9]` and reports:

- IoU (foreground and per-class mean)
- Pixel Accuracy
- Precision / Recall
- F1 Score
- Dice Score

Results are saved as `.txt` files and example visualizations (input | ground truth | prediction) are stored in the model directory.

---

## Baseline ST-UNet Module: `network/UNET_ST.py`

| Class / Function | Description |
|---|---|
| `DoubleConv` | Standard `Conv-BN-ReLU x2` block |
| `Down` | `MaxPool + DoubleConv` downsampling |
| `Up` | `Upsample + Concat + DoubleConv` upsampling |
| `SimpleContextAdd` | Attention module fusing temporal + spatial features |
| `UNetEncoder` | 4-stage UNet encoder (64→128→256→512→512) |
| `UNetDecoder` | 4-stage decoder with `SimpleContextAdd` at stages 1-3 |
| `UNet` | Standalone UNet (no temporal branch) |
| `STUNet` | Full baseline spatiotemporal model |
| `create_stunet_with_attention` | Factory function — recommended entry point |

---

## Proposed ST-UNet3+ Modules: 'network/STUnet3Plus.py'

| Class / Function | Description |
|---|---|
| `DoubleConv` | Standard `Conv-BN-ReLU x2` block |
| `Down` | `MaxPool + DoubleConv` downsampling |
| `Up` | `Upsample + Concat + DoubleConv` upsampling |
| `SimpleContextAdd` | Attention module fusing temporal + spatial features |
| `ST-UNet3+ Encoder` | 4-stage UNet encoder (64→128→256→512→512) |
| `ST-UNet3+ Decoder` | 4-stage decoder with `SimpleContextAdd` at stages 1-3 |
| `ST-UNet3+` | Full proposed spatiotemporal model |
| `create_unet3plus` | Factory function — recommended entry point |

## Citation

D. Bezborodov. Temporal feature fusion for wildfire segmentation in uav video. pages 1–5, 2025.
A. Shamsoshoara, F. Afghah, A. Razi, L. Zheng, P. Z. Ful´e, and E. Blasch. Aerial imagery pile burn detection using deep learning: the flame dataset. Computer Networks, page 108001, 2021.
K. Xu, L. Wen, G. Li, L. Bo, and Q. Huang. Spatiotemporal cnn for video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1379–1388, 2019. doi: 10.1109/CVPR.2019.00147.
