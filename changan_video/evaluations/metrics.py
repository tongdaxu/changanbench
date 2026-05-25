from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class VideoFrameMetric(ABC):
    """Base class for frame-wise metrics used by changan_video evaluation."""

    names: tuple[str, ...]

    @abstractmethod
    def __call__(self, reference, distorted) -> dict[str, float]:
        raise NotImplementedError


class PSNRFrameMetric(VideoFrameMetric):
    names = ("psnr",)

    def __init__(self, *, zero_mean: bool = False) -> None:
        self.zero_mean = zero_mean

    def __call__(self, reference, distorted) -> dict[str, float]:
        from changan_video.evaluations.psnr import get_psnr

        value = get_psnr(
            reference,
            distorted,
            zero_mean=self.zero_mean,
            is_video=False,
        )
        return {"psnr": _tensor_scalar(value)}


class SSIMFrameMetric(VideoFrameMetric):
    def __init__(
        self,
        *,
        include_ssim: bool = True,
        include_msssim: bool = True,
        zero_mean: bool = False,
    ) -> None:
        self.names = tuple(
            name
            for name, enabled in (
                ("ssim", include_ssim),
                ("msssim", include_msssim),
            )
            if enabled
        )
        self.zero_mean = zero_mean

    def __call__(self, reference, distorted) -> dict[str, float]:
        from changan_video.evaluations.ssim import get_ssim_and_msssim

        ssim_value, msssim_value = get_ssim_and_msssim(
            reference,
            distorted,
            zero_mean=self.zero_mean,
            is_video=False,
        )
        values = {}
        if "ssim" in self.names:
            values["ssim"] = _tensor_scalar(ssim_value)
        if "msssim" in self.names:
            values["msssim"] = _tensor_scalar(msssim_value)
        return values


class LPIPSFrameMetric(VideoFrameMetric):
    names = ("lpips",)

    def __init__(
        self,
        *,
        device: str,
        zero_mean: bool = False,
        network_type: str = "alex",
    ) -> None:
        if network_type not in {"alex", "vgg"}:
            raise ValueError("lpips network_type must be 'alex' or 'vgg'")
        self.zero_mean = zero_mean
        self.network_type = network_type
        from changan_video.evaluations.lpips import build_lpips_model

        self.loss_fn = build_lpips_model(network_type=network_type, device=device)

    def __call__(self, reference, distorted) -> dict[str, float]:
        from changan_video.evaluations.lpips import get_lpips_with_model

        value = get_lpips_with_model(
            reference,
            distorted,
            zero_mean=self.zero_mean,
            network_type=self.network_type,
            is_video=False,
            loss_fn=self.loss_fn,
        )
        return {"lpips": _tensor_scalar(value)}


def build_frame_metrics(
    metric_names: Sequence[str],
    *,
    device: str,
    zero_mean: bool,
    lpips_network: str,
) -> list[VideoFrameMetric]:
    names = tuple(metric_names)
    metrics: list[VideoFrameMetric] = []
    if "psnr" in names:
        metrics.append(PSNRFrameMetric(zero_mean=zero_mean))
    if "ssim" in names or "msssim" in names:
        metrics.append(
            SSIMFrameMetric(
                include_ssim="ssim" in names,
                include_msssim="msssim" in names,
                zero_mean=zero_mean,
            )
        )
    if "lpips" in names:
        metrics.append(
            LPIPSFrameMetric(
                device=device,
                zero_mean=zero_mean,
                network_type=lpips_network,
            )
        )
    return metrics


def _tensor_scalar(value) -> float:
    return float(value.detach().reshape(-1).mean().cpu().item())
