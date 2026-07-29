import torch


def _as_scalar_bpp(bpp, device):
    if isinstance(bpp, torch.Tensor):
        if bpp.numel() != 1:
            raise ValueError(
                "Patch codec must return exactly one BPP value for a single patch, "
                f"but got shape {tuple(bpp.shape)}."
            )
        return bpp.reshape(()).to(device=device, dtype=torch.float32)

    try:
        scalar = torch.as_tensor(bpp, dtype=torch.float32, device=device)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Patch codec BPP must be a Python scalar or a single-element Tensor."
        ) from exc
    if scalar.numel() != 1:
        raise ValueError(
            "Patch codec must return exactly one BPP value for a single patch, "
            f"but got shape {tuple(scalar.shape)}."
        )
    return scalar.reshape(())


def forward_image_in_patches(x, patch_size, forward_patch):
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("Codec input must be a Tensor with shape [B, C, H, W].")

    batch_size, _, height, width = x.shape
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("Codec input batch size, height, and width must be positive.")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Codec input height and width must both be divisible by {patch_size}, "
            f"but got H={height}, W={width}."
        )

    reconstructed_images = []
    image_bpps = []
    patch_pixels = patch_size * patch_size

    for image_index in range(batch_size):
        reconstructed_rows = []
        total_bits = None

        for top in range(0, height, patch_size):
            reconstructed_row = []
            for left in range(0, width, patch_size):
                patch = x[
                    image_index : image_index + 1,
                    :,
                    top : top + patch_size,
                    left : left + patch_size,
                ].contiguous()
                reconstructed_patch, patch_bpp = forward_patch(patch)

                if not isinstance(reconstructed_patch, torch.Tensor):
                    raise TypeError("Patch codec reconstruction must be a Tensor.")
                if reconstructed_patch.shape != patch.shape:
                    raise ValueError(
                        "Patch codec reconstruction shape must match its input patch: "
                        f"expected {tuple(patch.shape)}, got "
                        f"{tuple(reconstructed_patch.shape)}."
                    )

                patch_bpp = _as_scalar_bpp(
                    patch_bpp,
                    device=reconstructed_patch.device,
                )
                patch_bits = patch_bpp * patch_pixels
                total_bits = patch_bits if total_bits is None else total_bits + patch_bits
                reconstructed_row.append(reconstructed_patch)

            reconstructed_rows.append(torch.cat(reconstructed_row, dim=3))

        reconstructed_image = torch.cat(reconstructed_rows, dim=2)
        if reconstructed_image.shape != x[image_index : image_index + 1].shape:
            raise ValueError(
                "Reconstructed image shape must match the original codec input: "
                f"expected {tuple(x[image_index : image_index + 1].shape)}, got "
                f"{tuple(reconstructed_image.shape)}."
            )

        reconstructed_images.append(reconstructed_image)
        image_bpps.append(total_bits / (height * width))

    reconstruction = torch.cat(reconstructed_images, dim=0)
    bpp = torch.stack(image_bpps)
    return reconstruction, bpp
