from __future__ import annotations


def get_dists(x_input, x_recon, zero_mean=False, is_video=False):
    try:
        from DISTS_pytorch import DISTS
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DISTS was requested, but DISTS_pytorch is not installed."
        ) from exc

    if zero_mean:
        x_input = x_input * 0.5 + 0.5
        x_recon = x_recon * 0.5 + 0.5

    metric = DISTS().to(x_input.device)
    if is_video:
        from einops import rearrange

        batch = x_input.shape[0]
        x_input = rearrange(x_input, "b c t h w -> (b t) c h w")
        x_recon = rearrange(x_recon, "b c t h w -> (b t) c h w")
        values = metric(x_input, x_recon)
        values = rearrange(values, "(b t) -> b t", b=batch).mean(1)
        return values
    return metric(x_input, x_recon)
