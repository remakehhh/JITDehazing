"""
Gaussian Diffusion Process
Implements DDPM and DDIM sampling for image dehazing
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """
    Get beta schedule for diffusion process
    
    Args:
        beta_schedule: 'linear', 'cosine', or 'quad'
        beta_start: starting beta value
        beta_end: ending beta value
        num_diffusion_timesteps: number of diffusion steps
    
    Returns:
        betas: (T,) array of beta values
    """
    if beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'cosine':
        s = 0.008
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = np.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented")
    
    return betas


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Model for Image Dehazing
    Supports both DDPM and DDIM sampling
    """
    
    def __init__(
        self,
        model,
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000,
        beta_schedule='linear',
        model_var_type='fixedlarge',
        loss_type='l2',
    ):
        super().__init__()
        
        self.model = model
        self.num_timesteps = num_diffusion_timesteps
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        
        # Get beta schedule
        betas = get_beta_schedule(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps
        )
        
        betas = torch.from_numpy(betas).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]])))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: sample q(x_t | x_0)
        
        Args:
            x_start: (B, C, H, W) clean image
            t: (B,) timestep
            noise: (B, C, H, W) noise, if None, sample from N(0, I)
        
        Returns:
            x_t: (B, C, H, W) noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise
        
        Args:
            x_t: (B, C, H, W) noisy image
            t: (B,) timestep
            noise: (B, C, H, W) predicted noise
        
        Returns:
            x_0: (B, C, H, W) predicted clean image
        """
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_recip_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_recip_alphas_cumprod_t = sqrt_recip_alphas_cumprod_t.unsqueeze(-1)
            sqrt_recipm1_alphas_cumprod_t = sqrt_recipm1_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute mean and variance of posterior q(x_{t-1} | x_t, x_0)
        
        Args:
            x_start: (B, C, H, W) clean image
            x_t: (B, C, H, W) noisy image at timestep t
            t: (B,) timestep
        
        Returns:
            posterior_mean: (B, C, H, W)
            posterior_variance: (B,)
            posterior_log_variance: (B,)
        """
        posterior_mean_coef1 = self.posterior_mean_coef1[t]
        posterior_mean_coef2 = self.posterior_mean_coef2[t]
        
        # Reshape for broadcasting
        while len(posterior_mean_coef1.shape) < len(x_start.shape):
            posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
            posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)
        
        posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x_t
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance = self.posterior_log_variance_clipped[t]
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(self, x, t, clip_denoised=True):
        """
        Apply model to get p(x_{t-1} | x_t)
        
        Args:
            x: (B, C, H, W) noisy image
            t: (B,) timestep
            clip_denoised: whether to clip predicted x_0 to [-1, 1]
        
        Returns:
            model_mean: (B, C, H, W)
            posterior_variance: (B,)
            posterior_log_variance: (B,)
        """
        # Predict noise
        model_output = self.model(x, t)
        
        # Predict x_0
        x_recon = self.predict_start_from_noise(x, t, model_output)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)
        
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_recon, x, t)
        
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t) (DDPM sampling)
        
        Args:
            x: (B, C, H, W) noisy image at timestep t
            t: (B,) timestep
            clip_denoised: whether to clip predicted x_0 to [-1, 1]
        
        Returns:
            sample: (B, C, H, W) image at timestep t-1
        """
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample
    
    @torch.no_grad()
    def p_sample_loop(self, shape, device):
        """
        Generate samples using DDPM sampling
        
        Args:
            shape: shape of samples to generate
            device: device to generate samples on
        
        Returns:
            samples: (B, C, H, W) generated samples
        """
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
        
        return img
    
    @torch.no_grad()
    def ddim_sample(self, x, t, t_next, eta=0.0, clip_denoised=True):
        """
        Sample using DDIM
        
        Args:
            x: (B, C, H, W) noisy image at timestep t
            t: (B,) current timestep
            t_next: (B,) next timestep
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)
            clip_denoised: whether to clip predicted x_0 to [-1, 1]
        
        Returns:
            sample: (B, C, H, W) image at timestep t_next
        """
        # Predict noise
        model_output = self.model(x, t)
        
        # Predict x_0
        x_recon = self.predict_start_from_noise(x, t, model_output)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)
        
        # Get alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_t_next = self.alphas_cumprod[t_next] if t_next.min() >= 0 else torch.ones_like(alpha_t)
        
        # Reshape for broadcasting
        while len(alpha_t.shape) < len(x.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_t_next = alpha_t_next.unsqueeze(-1)
        
        # Compute variance
        sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next))
        
        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_next - sigma_t ** 2) * model_output
        
        # Sample
        noise = torch.randn_like(x)
        x_next = torch.sqrt(alpha_t_next) * x_recon + dir_xt + sigma_t * noise
        
        return x_next
    
    @torch.no_grad()
    def ddim_sample_loop(self, shape, device, ddim_timesteps=50, eta=0.0):
        """
        Generate samples using DDIM sampling
        
        Args:
            shape: shape of samples to generate
            device: device to generate samples on
            ddim_timesteps: number of DDIM sampling steps
            eta: DDIM eta parameter
        
        Returns:
            samples: (B, C, H, W) generated samples
        """
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        # Create sampling schedule
        step = self.num_timesteps // ddim_timesteps
        timesteps = np.arange(0, self.num_timesteps, step)
        timesteps = timesteps[::-1]  # Reverse order
        
        for i in tqdm(range(len(timesteps)), desc='DDIM Sampling'):
            t = torch.full((b,), timesteps[i], device=device, dtype=torch.long)
            t_next = torch.full((b,), timesteps[i+1], device=device, dtype=torch.long) \
                if i < len(timesteps) - 1 else torch.full((b,), -1, device=device, dtype=torch.long)
            
            img = self.ddim_sample(img, t, t_next, eta=eta)
        
        return img
    
    def training_losses(self, x_start, t, noise=None):
        """
        Compute training loss
        
        Args:
            x_start: (B, C, H, W) clean image
            t: (B,) timestep
            noise: (B, C, H, W) noise, if None, sample from N(0, I)
        
        Returns:
            loss: scalar loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise=noise)
        
        # Predict noise
        model_output = self.model(x_t, t)
        
        # Compute loss
        if self.loss_type == 'l1':
            loss = torch.abs(model_output - noise).mean()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(model_output, noise)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")
        
        return loss
    
    def forward(self, x, t, noise=None):
        """Forward pass for training"""
        return self.training_losses(x, t, noise=noise)
