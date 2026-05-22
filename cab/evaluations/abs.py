from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class MetricIface(nn.Module, ABC):
    """Abstract base class for all metrics.
    
    Metrics can be:
    - Stateless (PSNR, SSIM): compute per-batch
    - Stateful (FID): accumulate batch activations, compute at end
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x_input: torch.Tensor, x_recon: torch.Tensor, 
                zero_mean: bool = False, **kwargs) -> torch.Tensor:
        """Compute metric(s) for a batch.
        
        Args:
            x_input: Input images/videos, shape (B, C, H, W) or (B, C, T, H, W)
            x_recon: Reconstructed images/videos, same shape as x_input
            zero_mean: Whether input is in [-1, 1] range (True) or [0, 1] range (False)
            **kwargs: Additional metric-specific arguments
            
        Returns:
            Metric scores as torch.Tensor with shape (B,)
            For multi-output metrics, return tuple of tensors
        """
        raise NotImplementedError 