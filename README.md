# JiT-RSHazeDiff: Image Dehazing with JiT and Diffusion Models

A PyTorch implementation of image dehazing using the JiT (Joint image-Text) architecture combined with RSHazeDiff diffusion models.

## Overview

This project implements a complete image dehazing pipeline that combines:
- **JiT Architecture**: Vision Transformer with Rotary Position Embeddings (RoPE), QK-Norm attention, and SwiGLU FFN
- **RSHazeDiff**: Diffusion-based dehazing approach using DDPM/DDIM sampling
- **U-Net Backbone**: Multi-scale architecture with JiT attention blocks

## Features

- ✅ Complete JiT-UNet implementation with adaptive Layer Normalization (adaLN)
- ✅ DDPM and DDIM sampling support
- ✅ Exponential Moving Average (EMA) for stable training
- ✅ Multiple loss functions (L1, L2, Charbonnier, SSIM)
- ✅ Support for multiple datasets (DHID, LHID, RICE)
- ✅ Comprehensive evaluation metrics (PSNR, SSIM)
- ✅ TensorBoard logging
- ✅ Data augmentation
- ✅ Single and multi-GPU training support

## Architecture

### JiT Components

1. **RMSNorm**: Root Mean Square Layer Normalization for stable training
2. **VisionRotaryEmbeddingFast**: 2D rotary position embeddings for spatial awareness
3. **SwiGLU FFN**: Efficient feed-forward network with SwiGLU activation
4. **Attention with QK-Norm**: Multi-head attention with Query-Key normalization
5. **TimestepEmbedder**: Sinusoidal embeddings for diffusion timesteps
6. **JiTBlock**: Complete transformer block with adaLN modulation

### JiT-UNet Architecture

```
Input Image (3, 256, 256)
    ↓
Conv (3 → 128)
    ↓
Encoder (with JiT Attention at resolutions 32x32, 16x16)
    ├─ Level 1: 128 channels, 256x256
    ├─ Level 2: 256 channels, 128x128
    ├─ Level 3: 256 channels, 64x64
    └─ Level 4: 512 channels, 32x32
    ↓
Middle (with JiT Attention)
    ├─ ResBlock + JiTAttention + ResBlock
    ↓
Decoder (with Skip Connections & JiT Attention)
    ├─ Level 4: 512 channels, 32x32
    ├─ Level 3: 256 channels, 64x64
    ├─ Level 2: 256 channels, 128x128
    └─ Level 1: 128 channels, 256x256
    ↓
Output Conv (128 → 3)
    ↓
Dehazed Image (3, 256, 256)
```

### Diffusion Process

- **Forward Process**: Gradually add Gaussian noise to clean images
- **Reverse Process**: Denoise using JiT-UNet to recover clean images
- **DDPM Sampling**: Standard denoising diffusion probabilistic model
- **DDIM Sampling**: Faster deterministic sampling (10-50 steps vs 1000)

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- PyTorch 1.13+

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision numpy pillow pyyaml tqdm tensorboard scikit-image opencv-python einops
```

## Dataset Preparation

### Directory Structure

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── GT/          # Ground truth (clean) images
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── Haze/        # Hazy images
│       ├── img_001.png
│       ├── img_002.png
│       └── ...
├── val/
│   ├── GT/
│   └── Haze/
└── test/
    ├── GT/
    └── Haze/
```

### Supported Datasets

The code supports multiple dehazing datasets:

1. **DHID** (Dense-Haze Image Dataset)
2. **LHID** (Light-Haze Image Dataset)  
3. **RICE** (Realistic Image Contrast Enhancement)
4. Custom datasets following the above structure

### Dataset Naming Conventions

The dataloader automatically matches GT and Haze images by:
1. Exact filename match (recommended)
2. Common prefix/suffix patterns (e.g., `GT_xxx.png` ↔ `Haze_xxx.png`)
3. Sequential order (fallback)

## Configuration

Default configuration is in `configs/default.yaml`. Key parameters:

```yaml
model:
  ch: 128                    # Base channel dimension
  ch_mult: [1, 2, 2, 4]      # Channel multipliers per level
  num_res_blocks: 2          # Residual blocks per level
  attn_resolutions: [32, 16] # Where to apply attention
  num_heads: 8               # Attention heads

diffusion:
  num_diffusion_timesteps: 1000  # Training timesteps
  sampling_timesteps: 10         # DDIM inference steps
  beta_schedule: linear          # Noise schedule

data:
  image_size: 256
  batch_size: 8
  num_workers: 4

training:
  epochs: 200
  lr: 0.0002
  ema_rate: 0.9999
```

## Training

### Basic Training

```bash
python train.py \
    --config configs/default.yaml \
    --data_dir ./data/train \
    --output_dir ./checkpoints \
    --gpu 0
```

### Resume Training

```bash
python train.py \
    --config configs/default.yaml \
    --data_dir ./data/train \
    --output_dir ./checkpoints \
    --resume ./checkpoints/latest.pth \
    --gpu 0
```

### Custom Configuration

Create your own config file based on `configs/default.yaml` and specify it:

```bash
python train.py --config configs/my_config.yaml
```

### Multi-GPU Training

For multi-GPU training, set the `distributed` configuration in your YAML:

```yaml
distributed:
  enabled: true
  world_size: 4
```

Then run:

```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/default.yaml
```

### Training Outputs

- **Checkpoints**: Saved to `--output_dir` (default: `./checkpoints/`)
  - `latest.pth`: Latest checkpoint
  - `best.pth`: Best validation PSNR
  - `checkpoint_epoch_X.pth`: Periodic checkpoints
- **Logs**: TensorBoard logs in `./logs/` (default)
  - Training loss
  - Validation PSNR/SSIM
  - Learning rate

### Monitor Training

```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

## Evaluation

### Basic Evaluation

```bash
python eval.py \
    --config configs/default.yaml \
    --checkpoint ./checkpoints/best.pth \
    --data_dir ./data/test \
    --output_dir ./results \
    --save_images \
    --gpu 0
```

### Fast Sampling

Use fewer DDIM steps for faster inference:

```bash
python eval.py \
    --checkpoint ./checkpoints/best.pth \
    --data_dir ./data/test \
    --ddim_steps 10 \
    --ddim_eta 0.0 \
    --save_images
```

### Evaluation Arguments

- `--checkpoint`: Path to model checkpoint
- `--data_dir`: Test dataset directory
- `--output_dir`: Where to save results
- `--ddim_steps`: Number of sampling steps (default: 10, range: 1-1000)
- `--ddim_eta`: DDIM stochasticity (0=deterministic, 1=DDPM)
- `--save_images`: Save output images
- `--batch_size`: Batch size for evaluation

### Evaluation Outputs

Results are saved to `--output_dir`:
- `dehazed/`: Denoised images
- `hazy/`: Input hazy images
- `gt/`: Ground truth images
- `results.txt`: PSNR and SSIM metrics

## Model Details

### Key Implementation Details

#### JiTAttentionBlock

Combines multiple advanced techniques:

```python
1. RMSNorm for normalization
2. Adaptive Layer Norm (adaLN) modulation with timestep
3. QK-Normalized attention
4. Rotary Position Embeddings (RoPE)
5. SwiGLU feed-forward network
```

#### Diffusion Process

- **Training**: 1000 timesteps with linear beta schedule
- **Inference**: 10-50 DDIM steps for fast sampling
- **Noise Prediction**: Model predicts noise instead of image directly

### Memory Requirements

Approximate GPU memory for training:

| Image Size | Batch Size | GPU Memory |
|-----------|------------|------------|
| 256x256   | 8          | ~12 GB     |
| 256x256   | 4          | ~8 GB      |
| 512x512   | 4          | ~24 GB     |

## Results

Expected performance on standard datasets:

| Dataset | PSNR (dB) | SSIM |
|---------|-----------|------|
| DHID    | TBD       | TBD  |
| LHID    | TBD       | TBD  |
| RICE    | TBD       | TBD  |

*Note: Results depend on training epochs, dataset size, and hyperparameters*

## Tips and Best Practices

1. **Start Small**: Test with a small dataset first to ensure everything works
2. **Learning Rate**: Start with 2e-4, reduce if training is unstable
3. **Batch Size**: Larger is better, but limited by GPU memory
4. **DDIM Steps**: 10 steps is usually sufficient for evaluation
5. **EMA**: Always use EMA for better validation performance
6. **Data Augmentation**: Essential for good generalization
7. **Validation**: Monitor PSNR/SSIM during training

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size
- Reduce image size
- Use gradient checkpointing (not implemented)
- Use mixed precision training (not implemented)

### Poor Results

- Train longer (200+ epochs)
- Check dataset quality and alignment
- Verify GT and Haze images are correctly paired
- Increase model capacity (ch, ch_mult)
- Use data augmentation

### Slow Training

- Increase batch size if possible
- Use more workers for data loading
- Use DDIM with fewer steps for validation
- Reduce validation frequency

## Code Structure

```
JITDehazing/
├── models/
│   ├── __init__.py
│   ├── jit_blocks.py      # JiT components
│   ├── jit_unet.py        # JiT-UNet architecture
│   ├── diffusion.py       # Diffusion process
│   └── losses.py          # Loss functions
├── datasets/
│   ├── __init__.py
│   └── dehaze_dataset.py  # Dataset loader
├── utils/
│   ├── __init__.py
│   ├── ema.py             # EMA helper
│   └── metrics.py         # Evaluation metrics
├── configs/
│   └── default.yaml       # Default configuration
├── train.py               # Training script
├── eval.py                # Evaluation script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## References

- **JiT**: [Joint image-Text Transformer](https://github.com/LTH14/JiT)
- **RSHazeDiff**: [RSHazeDiff GitHub](https://github.com/jm-xiong/RSHazeDiff)
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **DDIM**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{jit-rshazediff,
  title={JiT-RSHazeDiff: Image Dehazing with JiT and Diffusion Models},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/remakehhh/JITDehazing}}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Acknowledgments

- JiT architecture from [LTH14/JiT](https://github.com/LTH14/JiT)
- RSHazeDiff from [jm-xiong/RSHazeDiff](https://github.com/jm-xiong/RSHazeDiff)
- Diffusion models community for DDPM/DDIM implementations

## Contact

For questions or issues, please open an issue on GitHub.