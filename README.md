# STCNN-FIRE

Dual-Branch Spatiotemporal Convolutional Neural Network - ST-UNet3+

## Quick start

Everything you need to train and evaluate is in [`stcnn/`](stcnn/):

```bash
cd stcnn/
pip install -r requirements.txt
python train.py --model stunet3plus --dataset fire --frame_nums 4 --output_dir <path_to_dir>
python test.py --checkpoint <path_to_training_output>
```

See [`stcnn/README.md`](stcnn/README.md) for full setup, dataset structure, and architecture details.

## Repository layout

```
stcnn/        ← self-contained: model, training, evaluation, dataloaders
```
