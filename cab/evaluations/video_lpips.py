import torch
import lpips as lpips_lib
from einops import rearrange
from cab.evaluations.abs import MetricIface


class LPIPSMetric(MetricIface):
    """LPIPS (Learned Perceptual Image Patch Similarity)"""

    def __init__(self, network_type: str = "alex", verbose: bool = False,
                 zero_mean: bool = False, is_video: bool = False, **kwargs):
        super().__init__()
        self.name = f"lpips_{network_type}"

        # Load the LPIPS model once during init
        self.loss_fn = lpips_lib.LPIPS(net=network_type, verbose=verbose).cuda()

    def forward(self, x_input: torch.Tensor, x_recon: torch.Tensor,
                zero_mean: bool = False, is_video: bool = False, **kwargs) -> torch.Tensor:
        """
        Args:
            x_input: Input in range [0, 1] or [-1, 1]
            x_recon: Reconstructed, same range
            zero_mean: True if range is [-1, 1], False if [0, 1]
            is_video: True if (B, C, T, H, W), False if (B, C, H, W)
        """
        
        # LPIPS expects [-1, 1] range
        if not zero_mean:
            x_input = x_input * 2 - 1
            x_recon = x_recon * 2 - 1
        
        with torch.no_grad():
            if is_video:
                # Process each frame separately, then average
                b = x_input.shape[0]
                x_input_flat = rearrange(x_input, "b c t h w -> (b t) c h w")
                x_recon_flat = rearrange(x_recon, "b c t h w -> (b t) c h w")
                
                lpips_scores = self.loss_fn(x_input_flat, x_recon_flat).squeeze()
                # Reshape back to (B, T) and average over time
                lpips_scores = rearrange(lpips_scores, "(b t) -> b t", b=b)
                return lpips_scores.mean(dim=1)
            else:
                lpips_scores = self.loss_fn(x_input, x_recon).squeeze()
                return lpips_scores


def get_lpips(x_input, x_recon, zero_mean=False, network_type="alex", is_video=False):
    """Legacy function interface"""
    metric = LPIPSMetric(network_type=network_type, zero_mean=zero_mean, is_video=is_video)
    return metric(x_input, x_recon, zero_mean=zero_mean, is_video=is_video)


def build_lpips_model(network_type="alex", device=None):
    assert network_type in ["alex", "vgg"]
    model = lpips_lib.LPIPS(net=network_type, verbose=False)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def get_lpips_with_model(
    x_input,
    x_recon,
    zero_mean=False,
    network_type="alex",
    is_video=False,
    loss_fn=None,
):
    assert network_type in ["alex", "vgg"]
    if loss_fn is None:
        loss_fn = build_lpips_model(network_type=network_type, device=x_input.device)
    if not zero_mean:
        x_input = x_input * 2 - 1
        x_recon = x_recon * 2 - 1
    if is_video:
        b = x_input.shape[0]
        x_input = rearrange(x_input, "b c t h w -> (b t) c h w")
        x_recon = rearrange(x_recon, "b c t h w -> (b t) c h w")
        d = loss_fn.forward(x_input, x_recon).squeeze()
        d = rearrange(d, "(b t) -> b t", b=b).mean(1)
    else:
        d = loss_fn.forward(x_input, x_recon)
    return d
