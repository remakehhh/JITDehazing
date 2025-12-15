"""
JiT-UNet Architecture
Implements U-Net with JiT attention blocks for image dehazing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .jit_blocks import (
    RMSNorm, VisionRotaryEmbeddingFast, SwiGLUFFN, 
    Attention, TimestepEmbedder, apply_rotary_emb
)


class ResnetBlock(nn.Module):
    """
    Residual Block with timestep embedding
    """
    
    def __init__(self, in_channels, out_channels=None, temb_channels=512, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return self.nin_shortcut(x) + h


class JiTAttentionBlock(nn.Module):
    """
    JiT Attention Block with RoPE and adaLN for U-Net
    Applies JiT-style attention to spatial features
    """
    
    def __init__(self, channels, num_heads=8, temb_channels=512):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Normalization layers
        self.norm1 = RMSNorm(channels)
        self.norm2 = RMSNorm(channels)
        
        # Attention with QK-Norm
        self.attn = Attention(
            channels, 
            num_heads=num_heads, 
            qk_norm=True,
            attn_drop=0.0,
            proj_drop=0.0
        )
        
        # SwiGLU FFN
        self.ffn = SwiGLUFFN(
            in_features=channels,
            hidden_features=channels * 4,
            drop=0.0
        )
        
        # adaLN modulation with timestep embedding
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_channels, 6 * channels)
        )
        
        # RoPE for positional encoding
        self.rope = VisionRotaryEmbeddingFast(self.head_dim // 2)
        
        # Input/output projections for spatial to sequence
        self.norm_in = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    
    def forward(self, x, temb):
        """
        Args:
            x: (B, C, H, W) spatial features
            temb: (B, temb_channels) timestep embedding
        """
        B, C, H, W = x.shape
        
        # Normalize input
        h = self.norm_in(x)
        
        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)
        
        # Get adaLN modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(temb).chunk(6, dim=1)
        
        # Attention block with adaLN-modulated normalization
        h_norm = self.norm1(h)
        h_modulated = h_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h_attn = self.attn(h_modulated, rope=self.rope)
        h = h + gate_msa.unsqueeze(1) * h_attn
        
        # FFN block with adaLN-modulated normalization
        h_norm = self.norm2(h)
        h_modulated = h_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h_ffn = self.ffn(h_modulated)
        h = h + gate_mlp.unsqueeze(1) * h_ffn
        
        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Residual connection
        return x + h


class Downsample(nn.Module):
    """Spatial downsampling layer"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling layer"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class JiTUNet(nn.Module):
    """
    JiT-UNet: U-Net with JiT attention blocks
    Combines U-Net architecture with JiT attention for image dehazing
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        ch=128,
        ch_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(32, 16),
        dropout=0.0,
        resolution=256,
        num_heads=8,
    ):
        super().__init__()
        
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        
        # Timestep embedding
        temb_ch = ch * 4
        self.temb = TimestepEmbedder(temb_ch)
        
        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                
                if curr_res in attn_resolutions:
                    attn.append(JiTAttentionBlock(
                        block_in,
                        num_heads=num_heads,
                        temb_channels=temb_ch
                    ))
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            
            self.down.append(down)
        
        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=temb_ch,
            dropout=dropout
        )
        self.mid.attn_1 = JiTAttentionBlock(
            block_in,
            num_heads=num_heads,
            temb_channels=temb_ch
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=temb_ch,
            dropout=dropout
        )
        
        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                
                block.append(ResnetBlock(
                    in_channels=block_in + skip_in,
                    out_channels=block_out,
                    temb_channels=temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                
                if curr_res in attn_resolutions:
                    attn.append(JiTAttentionBlock(
                        block_in,
                        num_heads=num_heads,
                        temb_channels=temb_ch
                    ))
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            
            self.up.insert(0, up)
        
        # Output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, t):
        """
        Args:
            x: (B, C, H, W) input noisy image
            t: (B,) timestep
        Returns:
            (B, C, H, W) predicted noise or denoised image
        """
        # Timestep embedding
        temb = self.temb(t)
        
        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h, temb)
                hs.append(h)
            
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h, temb)
        h = self.mid.block_2(h, temb)
        
        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = torch.cat([h, hs.pop()], dim=1)
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, temb)
            
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
