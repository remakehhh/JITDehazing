"""
Loss Functions for Image Dehazing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 smooth loss)
    More robust to outliers than L2 loss
    """
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted image
            target: ground truth image
        Returns:
            loss: scalar loss value
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss
    Measures perceptual similarity between images
    """
    
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)
    
    def gaussian(self, window_size, sigma=1.5):
        """Create Gaussian kernel"""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        """Create 2D Gaussian window"""
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2):
        """
        Calculate SSIM between two images
        
        Args:
            img1: first image
            img2: second image
        Returns:
            ssim_value: SSIM value
        """
        window = self.window.to(img1.device).type_as(img1)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted image
            target: ground truth image
        Returns:
            loss: 1 - SSIM (lower is better)
        """
        return 1 - self.ssim(pred, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features
    Measures high-level perceptual similarity
    """
    
    def __init__(self):
        super().__init__()
        # We'll use a simple L1 loss in feature space
        # In a full implementation, this would use pre-trained VGG features
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        """
        Simple perceptual loss (placeholder)
        In production, this should use VGG features
        """
        return self.criterion(pred, target)


class CombinedLoss(nn.Module):
    """
    Combined Loss Function
    Combines multiple loss functions with weights
    """
    
    def __init__(
        self,
        use_l1=True,
        use_l2=True,
        use_charbonnier=True,
        use_ssim=True,
        use_perceptual=False,
        l1_weight=1.0,
        l2_weight=1.0,
        charbonnier_weight=1.0,
        ssim_weight=1.0,
        perceptual_weight=0.1,
    ):
        super().__init__()
        
        self.use_l1 = use_l1
        self.use_l2 = use_l2
        self.use_charbonnier = use_charbonnier
        self.use_ssim = use_ssim
        self.use_perceptual = use_perceptual
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        
        if use_l1:
            self.l1_loss = nn.L1Loss()
        if use_l2:
            self.l2_loss = nn.MSELoss()
        if use_charbonnier:
            self.charbonnier_loss = CharbonnierLoss()
        if use_ssim:
            self.ssim_loss = SSIMLoss()
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted image
            target: ground truth image
        Returns:
            loss: combined loss value
            loss_dict: dictionary of individual losses
        """
        loss = 0.0
        loss_dict = {}
        
        if self.use_l1:
            l1_loss = self.l1_loss(pred, target)
            loss += self.l1_weight * l1_loss
            loss_dict['l1'] = l1_loss.item()
        
        if self.use_l2:
            l2_loss = self.l2_loss(pred, target)
            loss += self.l2_weight * l2_loss
            loss_dict['l2'] = l2_loss.item()
        
        if self.use_charbonnier:
            charbonnier_loss = self.charbonnier_loss(pred, target)
            loss += self.charbonnier_weight * charbonnier_loss
            loss_dict['charbonnier'] = charbonnier_loss.item()
        
        if self.use_ssim:
            ssim_loss = self.ssim_loss(pred, target)
            loss += self.ssim_weight * ssim_loss
            loss_dict['ssim'] = ssim_loss.item()
        
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(pred, target)
            loss += self.perceptual_weight * perceptual_loss
            loss_dict['perceptual'] = perceptual_loss.item()
        
        loss_dict['total'] = loss.item()
        
        return loss, loss_dict
