from abc import abstractmethod
from cab.complexity import params_m, gflops

import torch
import torch.nn as nn


def split_image_into_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split BCHW images into non-overlapping row-major patches.

    Returns a tensor with shape ``(B, N, C, patch_size, patch_size)``.
    No padding or cropping is performed.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected a BCHW tensor, got shape {tuple(x.shape)}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    batch, channels, height, width = x.shape
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Image height and width must be positive, got {height}x{width}."
        )
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Image size {height}x{width} is not divisible by "
            f"{patch_size}x{patch_size}; codec patching never pads or crops."
        )

    grid_h = height // patch_size
    grid_w = width // patch_size
    return (
        x.reshape(batch, channels, grid_h, patch_size, grid_w, patch_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, grid_h * grid_w, channels, patch_size, patch_size)
    )


def merge_image_patches(
    patches: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Reassemble row-major non-overlapping patches into BCHW images."""
    if patches.ndim != 5:
        raise ValueError(
            f"Expected a BNCHW patch tensor, got shape {tuple(patches.shape)}"
        )

    batch, patch_count, channels, patch_h, patch_w = patches.shape
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Output height and width must be positive, got {height}x{width}."
        )
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(
            f"Patch height and width must be positive, got {patch_h}x{patch_w}."
        )
    if height % patch_h or width % patch_w:
        raise ValueError(
            f"Output size {height}x{width} is not divisible by patch size "
            f"{patch_h}x{patch_w}."
        )

    grid_h = height // patch_h
    grid_w = width // patch_w
    expected_count = grid_h * grid_w
    if patch_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} patches for {height}x{width}, "
            f"got {patch_count}."
        )

    return (
        patches.reshape(batch, grid_h, grid_w, channels, patch_h, patch_w)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
    )


def patch_bits_to_image_bpp(
    patch_bits: torch.Tensor,
    batch_size: int,
    patches_per_image: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Aggregate encoded patch bit counts into one BPP value per image."""
    patch_bits = torch.as_tensor(patch_bits).reshape(-1)
    if batch_size <= 0 or patches_per_image <= 0:
        raise ValueError(
            "batch_size and patches_per_image must be positive, got "
            f"{batch_size} and {patches_per_image}."
        )
    expected_count = batch_size * patches_per_image
    if patch_bits.numel() != expected_count:
        raise ValueError(
            f"Expected {expected_count} patch bit counts, got "
            f"{patch_bits.numel()}."
        )

    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Image height and width must be positive, got {height}x{width}."
        )
    if not torch.isfinite(patch_bits).all():
        raise ValueError("Patch bit counts must all be finite.")
    if (patch_bits < 0).any():
        raise ValueError("Patch bit counts must not be negative.")
    return patch_bits.reshape(batch_size, patches_per_image).sum(dim=1) / (
        height * width
    )


class ImageCodecIface(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """Implement encoding/decoding.
        Args:
            x: (B, 3, H, W)
        Returns:
            xhat: (B, 3, H, W)
            bpp:  (B,) tensor
        """
        raise NotImplementedError
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.randn(1, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        return params_m(self)

    def decode_params_m(self):
        return params_m(self)

    def encode_time_ms(self, x, warmup=5, repeat=20):
        return None

    def decode_time_ms(self, x, warmup=5, repeat=20):
        return None

    def encode_gflops(self, x):
        return gflops(self, x)

    def decode_gflops(self, x):
        return None

class VideoCodecIface(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """Implement video encoding/decoding.
        Args:
            x: (B, 3, T, H, W)
        Returns:
            xhat: (B, 3, T, H, W)
            bpp:  (B,) tensor, bits per pixel per frame
        """
        raise NotImplementedError

    def fake_input(self, image_size=256, batch_size=1, frames=32, device=None, *args, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        frames = 32 if frames is None else int(frames)
        return torch.randn(
            int(batch_size), 3, frames, int(image_size), int(image_size),
            device=device,
        )

    def encode_params_m(self):
        return params_m(self)

    def decode_params_m(self):
        return params_m(self)

    def flops(self, x, *args, **kwargs):
        return None

    def encode_gflops(self, x):
        return self.flops(x)

    def decode_gflops(self, x):
        return None

    def param_count(self, x, *args, **kwargs):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

    def encode_time_ms(self, x, warmup=5, repeat=20):
        return None

    def decode_time_ms(self, x, warmup=5, repeat=20):
        return None

    def encode_time(self, x, *args, **kwargs):
        return self.encode_time_ms(x, *args, **kwargs)

    def decode_time(self, x, *args, **kwargs):
        return self.decode_time_ms(x, *args, **kwargs)

    # # complexity: to revise
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total params: {total_params:,}")
    # print(f"Trainable params: {trainable_params:,}")

    # # flops
    # import torch
    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # model.eval()
    # dummy = torch.randn(1, 3, 224, 224)
    # flops = FlopCountAnalysis(model, dummy)
    # print("Total FLOPs:", flops.total())
    # print(parameter_count_table(model))

    # # latency
    # import torch
    # import torch.utils.benchmark as benchmark
    # model.eval().cuda()
    # x = torch.randn(1, 3, 224, 224, device='cuda')
    # t = benchmark.Timer(
    # stmt='model(x)',
    # globals={'model': model, 'x': x},
    # )
    # result = t.blocked_autorange(min_run_time=1.0)
    # print(result)
