# Findings

This directory contains experimental scripts, intermediate results, and alternative architectures explored during the development of ST-UNet. These files are preserved for reproducibility and reference but are **not required** to run the main training and evaluation pipeline.

## Contents

| Folder | Contents |
|---|---|
| `experiments/` | Training and testing scripts for earlier model variants (ResUNet, DeepLab, PPM, CBAM, SwinUNet, etc.) |
| `networks/` | Alternative network architecture implementations |
| `dataloaders/` | Dataloaders for additional datasets (MSRA, VID, VOC) |
| `results/` | Evaluation `.txt` files and visualization images across model variants and thresholds |
| `environment/` | Conda environment export (`thesis-dependencies.txt`, `thesis-dependencies.yml`) |
| `third_party/` | `pyLucid` — LucidDream-based data augmentation tool for video sequences |

## Main Code

The production-ready code lives in [`../stcnn/`](../stcnn/):
- `stcnn/train.py` — training script
- `stcnn/test.py` — evaluation script
- `stcnn/network/UNET_ST.py` — core model architecture
