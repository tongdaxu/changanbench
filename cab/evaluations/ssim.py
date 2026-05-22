import torch
from pytorch_msssim import ssim, ms_ssim
from cab.evaluations.abs import MetricIface


class SSIMMetric(MetricIface):
    """SSIM and MS-SSIM"""

    def __init__(self, compute_msssim: bool = True,
                 zero_mean: bool = False, is_video: bool = False, **kwargs):
        super().__init__()
        self.compute_msssim = compute_msssim
        self.name = "ssim"

    @torch.no_grad()
    def forward(self, x_input: torch.Tensor, x_recon: torch.Tensor,
                zero_mean: bool = False, is_video: bool = False, **kwargs):
        """
        Returns:
            ssim_scores: (B,) tensor
            msssim_scores: (B,) tensor or None
        """

        # Convert to [0, 255]
        if zero_mean:
            x_input_0_255 = (x_input + 1) * 127.5
            x_recon_0_255 = (x_recon + 1) * 127.5
        else:
            x_input_0_255 = x_input * 255
            x_recon_0_255 = x_recon * 255

        # Check minimum size for MS-SSIM
        h_dim = 2 + is_video
        w_dim = 3 + is_video
        can_compute_msssim = (
            x_input_0_255.shape[h_dim] >= 256 and 
            x_input_0_255.shape[w_dim] >= 256 and
            self.compute_msssim
        )

        if is_video:
            ssim_scores = []
            msssim_scores = []
            
            for t in range(x_input_0_255.shape[2]):
                s = ssim(
                    x_input_0_255[:, :, t, :, :],
                    x_recon_0_255[:, :, t, :, :],
                    data_range=255,
                    size_average=False,
                )
                ssim_scores.append(s)
                
                if can_compute_msssim:
                    ms = ms_ssim(
                        x_input_0_255[:, :, t, :, :],
                        x_recon_0_255[:, :, t, :, :],
                        data_range=255,
                        size_average=False,
                    )
                    msssim_scores.append(ms)
            
            ssim_val = torch.stack(ssim_scores).mean(0)
            msssim_val = torch.stack(msssim_scores).mean(0) if can_compute_msssim else None
        else:
            ssim_val = ssim(x_input_0_255, x_recon_0_255, data_range=255, size_average=False)
            msssim_val = None
            
            if can_compute_msssim:
                msssim_val = ms_ssim(x_input_0_255, x_recon_0_255, data_range=255, size_average=False)

        if msssim_val is not None:
            return (ssim_val, msssim_val)
        else:
            return ssim_val


def get_ssim(x_input, x_recon, zero_mean=False, is_video=False):
    metric = SSIMMetric(compute_msssim=False)
    return metric(x_input, x_recon, zero_mean=zero_mean, is_video=is_video)


def get_ssim_and_msssim(x_input, x_recon, zero_mean=False, is_video=False):
    metric = SSIMMetric(compute_msssim=True)
    return metric(x_input, x_recon, zero_mean=zero_mean, is_video=is_video)