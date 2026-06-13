from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torchvision.utils as vutils


class VideoFrameMetricAdapter(torch.nn.Module):
    """Run an image metric frame-by-frame on a video batch."""

    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    @property
    def all_activations_x(self):
        return getattr(self.metric, "all_activations_x")

    @property
    def all_activations_xr(self):
        return getattr(self.metric, "all_activations_xr")

    def forward(self, x_input: torch.Tensor, x_recon: torch.Tensor, **kwargs):
        if x_input.ndim != 5:
            return self.metric(x_input, x_recon, **kwargs)

        batch_size, _, frame_count, _, _ = x_input.shape
        x_frames = video_to_image_batch(x_input)
        xr_frames = video_to_image_batch(x_recon)
        call_kwargs = dict(kwargs)
        call_kwargs["is_video"] = False
        try:
            out = self.metric(x_frames, xr_frames, **call_kwargs)
        except TypeError as exc:
            if "is_video" not in str(exc):
                raise
            call_kwargs.pop("is_video", None)
            out = self.metric(x_frames, xr_frames, **call_kwargs)
        return average_frame_outputs(out, batch_size, frame_count)


class FVDMetric:
    """Dataset-level FVD metric for the shared ddp_test loop."""

    def __init__(
        self,
        clip_length: int = 16,
        clip_stride: int = 16,
        model_path: str | None = None,
        device: str = "cuda",
    ):
        from cab.evaluations.fvd.video_fvd_score import get_i3d_model

        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride)
        self.model_path = model_path
        self.device = device
        self.model = get_i3d_model(model_path=model_path, device=device)
        self.reference_features = []
        self.distorted_features = []

    def __call__(self, x_input: torch.Tensor, x_recon: torch.Tensor, zero_mean: bool = False, **kwargs):
        if x_input.ndim != 5:
            raise ValueError("FVDMetric expects video tensors shaped (B, C, T, H, W)")
        if zero_mean:
            x_input = (x_input + 1.0) * 0.5
            x_recon = (x_recon + 1.0) * 0.5
        self.reference_features.append(self._extract(x_input))
        self.distorted_features.append(self._extract(x_recon))
        return None

    def gather_ddp_result(self, world_size: int):
        local_ref = _stack_or_empty(self.reference_features)
        local_dist = _stack_or_empty(self.distorted_features)
        gathered_ref = [None for _ in range(world_size)]
        gathered_dist = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_ref, local_ref)
        dist.all_gather_object(gathered_dist, local_dist)
        return gathered_ref, gathered_dist

    def compute_ddp_result(self, gathered) -> float:
        from cab.evaluations.fvd.video_fvd_score import compute_fvd_score

        reference, distorted = gathered
        reference = _concat_nonempty(reference)
        distorted = _concat_nonempty(distorted)
        if reference.size == 0 or distorted.size == 0:
            raise RuntimeError("No FVD features were accumulated.")
        return compute_fvd_score(distorted, reference)

    def _extract(self, video: torch.Tensor) -> np.ndarray:
        from cab.evaluations.fvd.video_fvd_score import extract_i3d_features

        video = video.detach().clamp(0.0, 1.0)
        arrays = (
            video.mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 4, 1)
            .cpu()
            .numpy()
        )
        clips = []
        for item in arrays:
            clips.extend(make_fvd_clips(item, self.clip_length, self.clip_stride))
        return extract_i3d_features(np.stack(clips, axis=0), model=self.model, device=self.device)


def adapt_metric_for_video(metric_name: str, metric):
    if metric_name in {"fvd", "vggt"}:
        return metric
    return VideoFrameMetricAdapter(metric)


def video_to_image_batch(video: torch.Tensor) -> torch.Tensor:
    return video.permute(0, 2, 1, 3, 4).reshape(
        video.shape[0] * video.shape[2],
        video.shape[1],
        video.shape[3],
        video.shape[4],
    )


def average_frame_outputs(out, batch_size: int, frame_count: int):
    if isinstance(out, tuple):
        return tuple(average_frame_outputs(item, batch_size, frame_count) for item in out)
    if isinstance(out, list):
        return [average_frame_outputs(item, batch_size, frame_count) for item in out]
    if out is None or not isinstance(out, torch.Tensor):
        return out

    values = out.reshape(-1)
    expected = batch_size * frame_count
    if values.numel() == expected:
        return values.reshape(batch_size, frame_count).mean(dim=1)
    if values.numel() == batch_size:
        return values
    if values.numel() == 1:
        return values.repeat(batch_size)
    if values.numel() % batch_size == 0:
        return values.reshape(batch_size, -1).mean(dim=1)
    return values


def save_reconstruction_preview(tensor: torch.Tensor, path: str) -> None:
    if tensor.ndim == 5:
        tensor = tensor[0, :, 0]
    elif tensor.ndim == 4 and tensor.shape[0] in {1, 3}:
        tensor = tensor[:, 0]
    vutils.save_image(tensor, path)


def gather_dataset_metric(metric, world_size: int):
    if hasattr(metric, "gather_ddp_result"):
        return metric.gather_ddp_result(world_size)
    return None


def print_dataset_metric(metric_name: str, metric, gathered) -> bool:
    if not hasattr(metric, "compute_ddp_result"):
        return False
    value = metric.compute_ddp_result(gathered)
    print(f"{metric_name:12s}: {value:.4f}")
    return True


def make_fvd_clips(video: np.ndarray, clip_length: int, clip_stride: int) -> list[np.ndarray]:
    if clip_length <= 0:
        raise ValueError("fvd clip_length must be greater than 0")
    if clip_stride <= 0:
        raise ValueError("fvd clip_stride must be greater than 0")

    total = video.shape[0]
    starts = list(range(0, max(total - clip_length + 1, 1), clip_stride))
    if starts[-1] != max(total - clip_length, 0):
        starts.append(max(total - clip_length, 0))

    clips = []
    for start in starts:
        clip = video[start : start + clip_length]
        if clip.shape[0] < clip_length:
            pad = np.repeat(clip[-1:, ...], clip_length - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        clips.append(clip)
    return clips


def _stack_or_empty(chunks) -> np.ndarray:
    if chunks:
        return np.vstack(chunks)
    return np.empty((0, 0), dtype=np.float32)


def _concat_nonempty(chunks) -> np.ndarray:
    arrays = [item for item in chunks if item is not None and item.size]
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(arrays)
