from DISTS_pytorch import DISTS as DISTSModel
from cab.evaluations.abs import MetricIface
from einops import rearrange
import torch


class DISTSMetric(MetricIface):
    """DISTS (Deep Image Structure and Texture Similarity)"""

    def __init__(self, zero_mean: bool = False, is_video: bool = False, **kwargs):
        super().__init__()
        self.name = "dists"
        self.model = DISTSModel().cuda()

    @torch.no_grad()
    def forward(self, x_input: torch.Tensor, x_recon: torch.Tensor,
                zero_mean: bool = False, is_video: bool = False, **kwargs) -> torch.Tensor:
        """
        Args:
            x_input: Input in range [0, 1] or [-1, 1]
            x_recon: Reconstructed, same range
            zero_mean: True if range is [-1, 1], False if [0, 1]
        """
        
        # DISTS expects [0, 1] range
        if zero_mean:
            x_input = x_input * 0.5 + 0.5
            x_recon = x_recon * 0.5 + 0.5
        
        if is_video:
            batch = x_input.shape[0]
            x_input = rearrange(x_input, "b c t h w -> (b t) c h w")
            x_recon = rearrange(x_recon, "b c t h w -> (b t) c h w")
            dists_scores = self.model(x_input, x_recon)
            return rearrange(dists_scores, "(b t) -> b t", b=batch).mean(1)

        return self.model(x_input, x_recon)


def get_dists(x_input, x_recon, zero_mean=False, is_video=False):
    """Legacy function interface"""
    metric = DISTSMetric()
    return metric(x_input, x_recon, zero_mean=zero_mean, is_video=is_video)
