"""
JiT (Joint image-Text) Core Components
Implements core building blocks for the JiT-RSHazeDiff fusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class VisionRotaryEmbeddingFast(nn.Module):
    """
    Fast Vision Rotary Position Embedding (RoPE)
    Applies 2D rotary position embeddings to vision features
    """
    
    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len if ft_seq_len is not None else pt_seq_len
        
        # Generate position indices
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
        
    def get_freqs(self, h, w, device):
        """Generate 2D frequency grid for height x width"""
        pos_h = torch.arange(h, device=device)
        pos_w = torch.arange(w, device=device)
        
        # Create 2D position grid
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        grid = torch.stack([grid_h, grid_w], dim=-1).float()
        
        # Compute frequencies
        freqs_h = torch.einsum('...i,j->...ij', grid[..., 0], self.freqs)
        freqs_w = torch.einsum('...i,j->...ij', grid[..., 1], self.freqs)
        
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        return freqs
    
    def forward(self, t, h, w):
        """
        Args:
            t: input tensor of shape (B, H, W, C)
            h: height
            w: width
        Returns:
            Tensor with rotary embeddings applied
        """
        freqs = self.get_freqs(h, w, t.device)
        
        # Split into real and imaginary parts
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        
        return freqs_cos, freqs_sin


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to input tensor"""
    # Split features into pairs
    x1, x2 = x.chunk(2, dim=-1)
    
    # Apply rotation
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)
    
    return rotated


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network
    Uses SwiGLU activation: SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.drop(self.w3(hidden))


class Attention(nn.Module):
    """
    Multi-head Attention with QK Normalization and optional RoPE
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, 
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, rope=None):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE if provided
        if rope is not None:
            # Compute spatial dimensions
            hw = int(math.sqrt(N))
            cos, sin = rope(q, hw, hw)
            # Reshape for application
            cos = cos.view(1, 1, hw, hw, -1).flatten(2, 3)
            sin = sin.view(1, 1, hw, hw, -1).flatten(2, 3)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Uses sinusoidal position embeddings.
    """
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: a 1-D Tensor of N indices, one per batch element.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.
        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class JiTBlock(nn.Module):
    """
    JiT Transformer Block with adaptive Layer Norm (adaLN) modulation
    Combines attention and feed-forward network with timestep conditioning
    """
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(
            hidden_size, 
            num_heads=num_heads, 
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
        
        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Initialize adaLN modulation to zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
        # RoPE for positional encoding
        self.rope = VisionRotaryEmbeddingFast(hidden_size // num_heads // 2)
    
    def forward(self, x, c):
        """
        Args:
            x: (B, N, C) input features
            c: (B, C) conditioning (e.g., timestep embedding)
        """
        # Get adaLN modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention block with adaLN
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_modulated, rope=self.rope)
        
        # MLP block with adaLN
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_modulated)
        
        return x
