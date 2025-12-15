"""
Evaluation Metrics for Image Dehazing
"""

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2, max_value=1.0):
    """
    Calculate PSNR between two images
    
    Args:
        img1: first image (H, W, C) or (C, H, W) numpy array or torch tensor
        img2: second image (H, W, C) or (C, H, W) numpy array or torch tensor
        max_value: maximum pixel value (1.0 for normalized images, 255 for uint8)
    
    Returns:
        psnr_value: PSNR in dB
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Convert from (C, H, W) to (H, W, C) if needed
    if img1.ndim == 3 and img1.shape[0] in [1, 3]:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] in [1, 3]:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Calculate PSNR
    return psnr(img1, img2, data_range=max_value)


def calculate_ssim(img1, img2, max_value=1.0):
    """
    Calculate SSIM between two images
    
    Args:
        img1: first image (H, W, C) or (C, H, W) numpy array or torch tensor
        img2: second image (H, W, C) or (C, H, W) numpy array or torch tensor
        max_value: maximum pixel value (1.0 for normalized images, 255 for uint8)
    
    Returns:
        ssim_value: SSIM value (0-1)
    """
    # Convert to numpy if torch tensor
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Convert from (C, H, W) to (H, W, C) if needed
    if img1.ndim == 3 and img1.shape[0] in [1, 3]:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] in [1, 3]:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Calculate SSIM
    if img1.ndim == 3:
        return ssim(img1, img2, data_range=max_value, channel_axis=2)
    else:
        return ssim(img1, img2, data_range=max_value)


def batch_psnr(img1, img2, max_value=1.0):
    """
    Calculate PSNR for a batch of images
    
    Args:
        img1: first batch of images (B, C, H, W)
        img2: second batch of images (B, C, H, W)
        max_value: maximum pixel value
    
    Returns:
        psnr_values: list of PSNR values
    """
    psnr_values = []
    for i in range(img1.shape[0]):
        psnr_val = calculate_psnr(img1[i], img2[i], max_value)
        psnr_values.append(psnr_val)
    return psnr_values


def batch_ssim(img1, img2, max_value=1.0):
    """
    Calculate SSIM for a batch of images
    
    Args:
        img1: first batch of images (B, C, H, W)
        img2: second batch of images (B, C, H, W)
        max_value: maximum pixel value
    
    Returns:
        ssim_values: list of SSIM values
    """
    ssim_values = []
    for i in range(img1.shape[0]):
        ssim_val = calculate_ssim(img1[i], img2[i], max_value)
        ssim_values.append(ssim_val)
    return ssim_values
