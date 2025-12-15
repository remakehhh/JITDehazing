"""
Training Script for JiT-RSHazeDiff Model
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models import JiTUNet, GaussianDiffusion
from datasets import DehazeDataset
from utils import EMAHelper, calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(description='Train JiT-RSHazeDiff model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset directory (overrides config)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, ema_helper, epoch, step, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema_helper.state_dict() if ema_helper else None,
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None, ema_helper=None):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if ema_helper and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
        ema_helper.load_state_dict(checkpoint['ema_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    
    print(f"Checkpoint loaded from {path} (epoch {epoch}, step {step})")
    return epoch, step


def train_epoch(model, diffusion, dataloader, optimizer, device, epoch, writer, config, ema_helper=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    log_interval = config['training']['log_interval']
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for i, batch in enumerate(pbar):
        # Move data to device
        gt = batch['gt'].to(device)
        
        # Sample timesteps
        t = torch.randint(0, diffusion.num_timesteps, (gt.shape[0],), device=device)
        
        # Compute loss
        loss = diffusion.training_losses(gt, t)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if 'grad_clip' in config['training']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        # Update EMA
        if ema_helper:
            ema_helper.update(model)
        
        # Logging
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # TensorBoard logging
        step = epoch * len(dataloader) + i
        if step % log_interval == 0:
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(model, diffusion, dataloader, device, epoch, writer, config):
    """Validate the model"""
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    # Use DDIM for faster sampling during validation
    ddim_steps = config['diffusion'].get('sampling_timesteps', 10)
    
    pbar = tqdm(dataloader, desc='Validation')
    for i, batch in enumerate(pbar):
        gt = batch['gt'].to(device)
        haze = batch['haze'].to(device)
        
        # Sample using DDIM
        samples = diffusion.ddim_sample_loop(
            haze.shape, 
            device, 
            ddim_timesteps=ddim_steps,
            eta=0.0
        )
        
        # Denormalize images from [-1, 1] to [0, 1]
        samples_denorm = (samples + 1.0) / 2.0
        gt_denorm = (gt + 1.0) / 2.0
        
        # Calculate metrics
        for j in range(gt.shape[0]):
            psnr = calculate_psnr(samples_denorm[j], gt_denorm[j], max_value=1.0)
            ssim = calculate_ssim(samples_denorm[j], gt_denorm[j], max_value=1.0)
            
            total_psnr += psnr
            total_ssim += ssim
            num_samples += 1
        
        pbar.set_postfix({
            'psnr': f'{total_psnr/num_samples:.2f}',
            'ssim': f'{total_ssim/num_samples:.4f}'
        })
        
        # Limit validation samples for speed
        if i >= 10:  # Only validate on first 10 batches
            break
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    # Log to TensorBoard
    writer.add_scalar('val/psnr', avg_psnr, epoch)
    writer.add_scalar('val/ssim', avg_ssim, epoch)
    
    print(f"Validation - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim


def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['train_dir'] = args.data_dir
    
    # Set seed
    set_seed(args.seed if args.seed else config['training']['seed'])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DehazeDataset(
        root_dir=config['data']['train_dir'],
        image_size=config['data']['image_size'],
        split='train',
        augment=config['data']['augment']
    )
    
    val_dataset = DehazeDataset(
        root_dir=config['data'].get('val_dir', config['data']['train_dir']),
        image_size=config['data']['image_size'],
        split='val',
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    unet = JiTUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        ch=config['model']['ch'],
        ch_mult=config['model']['ch_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attn_resolutions=config['model']['attn_resolutions'],
        dropout=config['model']['dropout'],
        resolution=config['model']['resolution'],
        num_heads=config['model']['num_heads'],
    ).to(device)
    
    # Create diffusion model
    diffusion = GaussianDiffusion(
        model=unet,
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        num_diffusion_timesteps=config['diffusion']['num_diffusion_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        model_var_type=config['diffusion']['model_var_type'],
        loss_type=config['diffusion']['loss_type'],
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        diffusion.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create EMA helper
    ema_helper = None
    if config['training'].get('use_ema', True):
        ema_helper = EMAHelper(mu=config['training']['ema_rate'])
        ema_helper.register(unet)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, unet, optimizer, ema_helper)
        start_epoch += 1
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=config['logging']['log_dir'])
    
    # Training loop
    print("Starting training...")
    best_psnr = 0.0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        avg_loss = train_epoch(
            unet, diffusion, train_loader, optimizer, device, 
            epoch, writer, config, ema_helper
        )
        
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # Validate
        if (epoch + 1) % config['training']['val_interval'] == 0:
            avg_psnr, avg_ssim = validate(
                unet, diffusion, val_loader, device, epoch, writer, config
            )
            
            # Save best model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_checkpoint(
                    unet, optimizer, ema_helper, epoch, 0,
                    os.path.join(args.output_dir, 'best.pth')
                )
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_checkpoint(
                unet, optimizer, ema_helper, epoch, 0,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
        
        # Save latest checkpoint
        save_checkpoint(
            unet, optimizer, ema_helper, epoch, 0,
            os.path.join(args.output_dir, 'latest.pth')
        )
    
    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()
