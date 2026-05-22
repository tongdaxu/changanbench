import torch
from cab.evaluations.abs import MetricIface


class PSNRMetric(MetricIface):
    """Peak Signal to Noise Ratio"""

    def __init__(self, zero_mean: bool = False, is_video: bool = False, **kwargs):
        super().__init__()
        self.name = "psnr"

    @torch.no_grad()
    def forward(self, x_input: torch.Tensor, x_recon: torch.Tensor, 
                zero_mean: bool = False, is_video: bool = False, **kwargs) -> torch.Tensor:
        """
        Args:
            x_input: Input in range [0, 1] or [-1, 1] depending on zero_mean
            x_recon: Reconstructed, same range
            zero_mean: True if range is [-1, 1], False if [0, 1]
            is_video: True if input is (B, C, T, H, W), False if (B, C, H, W)
        """

        # Convert to [0, 255] range
        if zero_mean:
            x_input_0_255 = (x_input + 1) * 127.5
            x_recon_0_255 = (x_recon + 1) * 127.5
        else:
            x_input_0_255 = x_input * 255
            x_recon_0_255 = x_recon * 255

        # Calculate MSE
        if is_video:
            dim = [1, 2, 3, 4]  # (B, C, T, H, W)
        else:
            dim = [1, 2, 3]     # (B, C, H, W)
        
        mse = torch.mean((x_input_0_255 - x_recon_0_255) ** 2, dim=dim)
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        return psnr


def get_psnr(x_input, x_recon, zero_mean=False, is_video=False):
    """Legacy function interface"""
    metric = PSNRMetric(zero_mean=zero_mean, is_video=is_video)
    return metric(x_input, x_recon, zero_mean=zero_mean, is_video=is_video)