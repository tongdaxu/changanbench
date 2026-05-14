def build_lpips_model(network_type="alex", device=None):
    """Build one reusable LPIPS model for frame-by-frame evaluation."""

    assert network_type in ["alex", "vgg"]
    try:
        import lpips
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LPIPS was requested, but the 'lpips' package is not installed. "
            "Install lpips or remove lpips from --metrics."
        ) from exc

    model = lpips.LPIPS(net=network_type, verbose=False)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def get_lpips(x_input, x_recon, zero_mean=False, network_type="alex", is_video=False):
    assert network_type in ["alex", "vgg"]
    loss_fn = build_lpips_model(network_type=network_type, device=x_input.device)
    return get_lpips_with_model(
        x_input,
        x_recon,
        zero_mean=zero_mean,
        network_type=network_type,
        is_video=is_video,
        loss_fn=loss_fn,
    )


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
        from einops import rearrange

        b = x_input.shape[0]
        x_input = rearrange(x_input, "b c t h w -> (b t) c h w")
        x_recon = rearrange(x_recon, "b c t h w -> (b t) c h w")
        d = loss_fn.forward(x_input, x_recon).squeeze()
        d = rearrange(d, "(b t) -> b t", b=b).mean(1)
    else:
        d = loss_fn.forward(x_input, x_recon)
    return d
