# STCNN-FIRE

Spatio-Temporal CNN for fire segmentation in video sequences.

## Quick start

Everything you need to train and evaluate is in [`stcnn/`](stcnn/):

```bash
cd stcnn/
pip install -r requirements.txt
python train.py --dataset fire --frame_nums 4
python test.py
```

See [`stcnn/README.md`](stcnn/README.md) for full setup, dataset structure, and architecture details.

## Repository layout

```
stcnn/        ← self-contained: model, training, evaluation, dataloaders
findings/     ← experimental scripts, ablation results, earlier architectures
```
