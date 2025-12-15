"""
Evaluation Script for JiT-RSHazeDiff Model
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from models import JiTUNet, GaussianDiffusion
from datasets import DehazeDataset
from utils import calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate JiT-RSHazeDiff model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--ddim_steps', type=int, default=10,
                       help='Number of DDIM sampling steps')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                       help='DDIM eta parameter (0=deterministic)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID to use')
    parser.add_argument('--save_images', action='store_true',
                       help='Save output images')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_image(tensor, path):
    """
    Save tensor as image
    
    Args:
        tensor: (C, H, W) tensor in range [-1, 1]
        path: output file path
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # Convert to numpy and scale to [0, 255]
    img = tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    img = (img * 255).astype(np.uint8)
    
    # Save image
    Image.fromarray(img).save(path)


def load_checkpoint(path, model):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
        # Use EMA weights if available
        model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Checkpoint loaded from {path}")


@torch.no_grad()
def evaluate(model, diffusion, dataloader, device, args):
    """Evaluate the model on test set"""
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    # Create output directory
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'dehazed'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'hazy'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'gt'), exist_ok=True)
    
    pbar = tqdm(dataloader, desc='Evaluation')
    for i, batch in enumerate(pbar):
        gt = batch['gt'].to(device)
        haze = batch['haze'].to(device)
        
        # Sample using DDIM
        samples = diffusion.ddim_sample_loop(
            haze.shape,
            device,
            ddim_timesteps=args.ddim_steps,
            eta=args.ddim_eta
        )
        
        # Denormalize images from [-1, 1] to [0, 1]
        samples_denorm = (samples + 1.0) / 2.0
        gt_denorm = (gt + 1.0) / 2.0
        
        # Calculate metrics for each image in batch
        for j in range(gt.shape[0]):
            psnr = calculate_psnr(samples_denorm[j], gt_denorm[j], max_value=1.0)
            ssim = calculate_ssim(samples_denorm[j], gt_denorm[j], max_value=1.0)
            
            total_psnr += psnr
            total_ssim += ssim
            num_samples += 1
            
            # Save images
            if args.save_images:
                img_idx = i * args.batch_size + j
                save_image(samples[j], os.path.join(args.output_dir, 'dehazed', f'{img_idx:04d}.png'))
                save_image(haze[j], os.path.join(args.output_dir, 'hazy', f'{img_idx:04d}.png'))
                save_image(gt[j], os.path.join(args.output_dir, 'gt', f'{img_idx:04d}.png'))
        
        # Update progress bar
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        pbar.set_postfix({
            'psnr': f'{avg_psnr:.2f}',
            'ssim': f'{avg_ssim:.4f}'
        })
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return avg_psnr, avg_ssim, num_samples


def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Create dataset
    print("Loading dataset...")
    test_dataset = DehazeDataset(
        root_dir=args.data_dir,
        image_size=config['data']['image_size'],
        split='test',
        augment=False
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
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
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, unet)
    
    # Evaluate
    print("Starting evaluation...")
    print(f"Using DDIM with {args.ddim_steps} steps (eta={args.ddim_eta})")
    
    avg_psnr, avg_ssim, num_samples = evaluate(
        unet, diffusion, test_loader, device, args
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Number of samples: {num_samples}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("="*50)
    
    if args.save_images:
        print(f"\nResults saved to {args.output_dir}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"DDIM steps: {args.ddim_steps}\n")
        f.write(f"DDIM eta: {args.ddim_eta}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
