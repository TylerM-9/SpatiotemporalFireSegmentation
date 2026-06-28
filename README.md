# ST-UNet: Spatio-Temporal UNet for Fire Segmentation

A spatio-temporal segmentation network that combines a **UNet encoder/decoder** with a **temporal prediction branch** to detect and segment fire in video sequences.

The key idea: a pretrained temporal branch processes previous frames to predict motion context, and an attention module (`SimpleContextAdd`) fuses this temporal signal into the spatial segmentation decoder at multiple scales.

---

## Architecture Overview

```
Input: frame sequence [T frames] + current frame
           |                          |
   Temporal Branch                Spatial Branch
   (FramePredEncoder)             (UNetEncoder)
         |                              |
   (FramePredDecoder)               [64, 128, 256, 512, 512]
    pred_features                         |
   [512, 256, 64]               UNetDecoder w/ SimpleContextAdd
         |__________________________|
                     |
              Segmentation output
                 [B, 1, H, W]
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
│   ├── UNET_ST.py             # Core model: UNetEncoder, UNetDecoder, STUNet, SimpleContextAdd
│   ├── joint_pred_seg.py      # Temporal branch: FramePredEncoder, FramePredDecoder
│   ├── googlenet.py           # Inception-v3 discriminator (GAN training)
│   └── shuffle.py             # ShuffleNetV2 blocks (used by temporal branch)
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
├── requirements.txt
└── findings/                  # Experimental scripts, ablation results, earlier architectures
```

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/BezboDima/STCNN_FIRE.git
cd STCNN_FIRE
pip install -r requirements.txt
```

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

The temporal branch (`FramePredEncoder` / `FramePredDecoder`) must be initialized from pretrained frame-prediction weights before training the full ST-UNet.

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
| `--dataset` | `fire` | `fire` (train+val) or `davis` (train only) |
| `--frame_nums` | `4` | Number of temporal context frames |
| `--resume_epoch` | `0` | Epoch to resume from (0 = fresh start) |
| `--pretrained_seg` | `None` | Path to pretrained segmentation checkpoint |

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

## Core Module: `network/UNET_ST.py`

The file contains all building blocks researchers may want to extend:

| Class / Function | Description |
|---|---|
| `DoubleConv` | Standard `Conv-BN-ReLU x2` block |
| `Down` | `MaxPool + DoubleConv` downsampling |
| `Up` | `Upsample + Concat + DoubleConv` upsampling |
| `SimpleContextAdd` | Attention module fusing temporal + spatial features |
| `UNetEncoder` | 4-stage UNet encoder (64→128→256→512→512) |
| `UNetDecoder` | 4-stage decoder with `SimpleContextAdd` at stages 1-3 |
| `UNet` | Standalone UNet (no temporal branch) |
| `STUNet` | Full spatio-temporal model |
| `create_stunet_with_attention` | Factory function — recommended entry point |

Run `python network/UNET_ST.py` to execute built-in architecture tests.

---

## Findings

The `findings/` directory contains earlier experiments, ablations, and alternative architectures explored during development:

- `findings/experiments/` — training and testing scripts for other model variants
- `findings/networks/` — alternative architectures (DeepLab, ResUNet, CBAM, SwinUNet, MobileNetV2)
- `findings/dataloaders/` — dataloaders for MSRA, VID, VOC datasets
- `findings/results/` — evaluation results and visualizations across model variants
- `findings/environment/` — conda environment specification
- `findings/third_party/` — `pyLucid` (LucidDream data augmentation tool)

---

## Citation

If you use this code, please cite the relevant work. The temporal prediction branch is based on a video frame prediction network trained adversarially with an Inception-v3 discriminator. The spatial branch follows the standard UNet architecture.
