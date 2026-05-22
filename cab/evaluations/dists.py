import torch
from DISTS_pytorch import DISTS as DISTSModel
from cab.evaluations.abs import MetricIface


class DISTSMetric(MetricIface):
    """DISTS (Deep Image Structure and Texture Similarity)"""

    def __init__(self, zero_mean: bool = False, **kwargs):
        super().__init__()
        self.name = "dists"
        self.model = DISTSModel()

    @torch.no_grad()
    def forward(self, x_input: torch.Tensor, x_recon: torch.Tensor,
                zero_mean: bool = False, **kwargs) -> torch.Tensor:
        """
        Args:
            x_input: Input in range [0, 1] or [-1, 1]
            x_recon: Reconstructed, same range
            zero_mean: True if range is [-1, 1], False if [0, 1]
        """
        # Move model to same device
        self.model = self.model.to(x_input.device)
        
        # DISTS expects [0, 1] range
        if zero_mean:
            x_input = x_input * 0.5 + 0.5
            x_recon = x_recon * 0.5 + 0.5
        
        dists_scores = self.model(x_input, x_recon)
        return dists_scores


def get_dists(x_input, x_recon, zero_mean=False):
    """Legacy function interface"""
    metric = DISTSMetric()
    return metric(x_input, x_recon, zero_mean=zero_mean)