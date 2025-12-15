"""
Exponential Moving Average (EMA) Helper
"""

import torch


class EMAHelper:
    """
    Exponential Moving Average for model parameters
    Helps stabilize training and improve performance
    """
    
    def __init__(self, mu=0.9999):
        """
        Args:
            mu: decay rate for EMA
        """
        self.mu = mu
        self.shadow = {}
    
    def register(self, module):
        """
        Register model parameters for EMA tracking
        
        Args:
            module: PyTorch module to track
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, module):
        """
        Update EMA parameters
        
        Args:
            module: PyTorch module with current parameters
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def ema(self, module):
        """
        Apply EMA parameters to module (for evaluation)
        
        Args:
            module: PyTorch module to apply EMA to
        """
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data.copy_(self.shadow[name])
    
    def ema_copy(self, module):
        """
        Create a copy of module with EMA parameters
        
        Args:
            module: PyTorch module to copy
        Returns:
            module_copy: copy with EMA parameters
        """
        module_copy = type(module)(module.config).to(module.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy
    
    def state_dict(self):
        """
        Get state dict for checkpointing
        
        Returns:
            state dict with EMA parameters
        """
        return self.shadow
    
    def load_state_dict(self, state_dict):
        """
        Load state dict from checkpoint
        
        Args:
            state_dict: saved EMA parameters
        """
        self.shadow = state_dict
