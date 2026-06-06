from __future__ import annotations

import math
from typing import Sequence

from cab.evaluations.video_metrics import build_frame_metrics
from cab.evaluations.video_types import FrameMetricResult, MetricSummary


class VideoFrameMetricRunner:
    def __init__(
        self,
        *,
        metric_names: Sequence[str],
        device: str,
        zero_mean: bool,
        lpips_network: str,
    ) -> None:
        if lpips_network not in {"alex", "vgg"}:
            raise ValueError("lpips_network must be 'alex' or 'vgg'")

        self.metric_names = tuple(metric_names)
        self.zero_mean = zero_mean
        self.lpips_network = lpips_network
        self.torch = _import_torch()
        self.device = resolve_torch_device(device)
        self.metrics = build_frame_metrics(
            self.metric_names,
            device=self.device,
            zero_mean=self.zero_mean,
            lpips_network=self.lpips_network,
        )

    def score(self, reference, distorted) -> dict[str, float]:
        reference_tensor = self._array_to_tensor(reference)
        distorted_tensor = self._array_to_tensor(distorted)
        if self.zero_mean:
            reference_tensor = reference_tensor * 2 - 1
            distorted_tensor = distorted_tensor * 2 - 1

        values: dict[str, float] = {}
        with self.torch.inference_mode():
            for metric in self.metrics:
                values.update(metric(reference_tensor, distorted_tensor))

        return {name: values[name] for name in self.metric_names}

    def _array_to_tensor(self, array):
        import numpy as np

        if array.dtype != np.uint8:
            raise TypeError(f"Expected uint8 frame, got {array.dtype}")
        tensor = self.torch.from_numpy(np.ascontiguousarray(array))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=self.device, dtype=self.torch.float32).div_(255.0)


class FidFeatureRunner:
    def __init__(self, *, device: str) -> None:
        self.torch = _import_torch()
        self.device = resolve_torch_device(device)

        from cab.evaluations.fid.inception import InceptionV3

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(self.device)
        self.model.eval()

    def extract(self, frame):
        import numpy as np

        tensor = self.torch.from_numpy(np.ascontiguousarray(frame))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.torch.float32).div_(255.0)
        with self.torch.inference_mode():
            pred = self.model(tensor)[0]
        return pred.squeeze(3).squeeze(2).detach().cpu().numpy()


def fid_summary(reference_features, distorted_features) -> MetricSummary:
    import numpy as np

    from cab.evaluations.fid.video_fid_score import compute_fid_score

    reference = np.concatenate(reference_features, axis=0)
    distorted = np.concatenate(distorted_features, axis=0)
    value = compute_fid_score(distorted, reference)
    return scalar_summary("fid", value, count=int(distorted.shape[0]))


def fvd_summary(
    reference_frames,
    distorted_frames,
    *,
    device: str,
    clip_length: int,
    clip_stride: int,
    model_path: str | None,
) -> MetricSummary:
    from cab.evaluations.fvd.video_fvd_score import (
        compute_fvd_score,
        extract_i3d_features,
        get_i3d_model,
    )

    reference_clips = _make_fvd_clips(reference_frames, clip_length, clip_stride)
    distorted_clips = _make_fvd_clips(distorted_frames, clip_length, clip_stride)
    model = get_i3d_model(model_path=model_path, device=device)
    reference_features = extract_i3d_features(reference_clips, model=model, device=device)
    distorted_features = extract_i3d_features(distorted_clips, model=model, device=device)
    value = compute_fvd_score(distorted_features, reference_features)
    return scalar_summary("fvd", value, count=int(distorted_features.shape[0]))


def summarize_frame_metrics(
    frame_metrics: Sequence[FrameMetricResult],
    metric_names: Sequence[str],
) -> dict[str, MetricSummary]:
    import numpy as np

    summaries: dict[str, MetricSummary] = {}
    for name in metric_names:
        values = np.array([item.values[name] for item in frame_metrics], dtype=np.float64)
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            summaries[name] = MetricSummary(
                name=name,
                count=0,
                mean=math.nan,
                std=math.nan,
                min=math.nan,
                max=math.nan,
            )
            continue

        summaries[name] = MetricSummary(
            name=name,
            count=int(valid.size),
            mean=float(np.mean(valid)),
            std=float(np.std(valid)),
            min=float(np.min(valid)),
            max=float(np.max(valid)),
        )
    return summaries


def scalar_summary(name: str, value: float, count: int) -> MetricSummary:
    return MetricSummary(
        name=name,
        count=count,
        mean=float(value),
        std=0.0,
        min=float(value),
        max=float(value),
    )


def resolve_torch_device(device: str) -> str:
    torch = _import_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _make_fvd_clips(frames, clip_length: int, clip_stride: int):
    import numpy as np

    if clip_length <= 0:
        raise ValueError("fvd_clip_length must be greater than 0")
    if clip_stride <= 0:
        raise ValueError("fvd_clip_stride must be greater than 0")
    if not frames:
        raise ValueError("No frames available for FVD")

    clips = []
    total = len(frames)
    starts = list(range(0, max(total - clip_length + 1, 1), clip_stride))
    if starts[-1] != max(total - clip_length, 0):
        starts.append(max(total - clip_length, 0))

    for start in starts:
        clip = frames[start : start + clip_length]
        if len(clip) < clip_length:
            clip = [*clip, *([clip[-1]] * (clip_length - len(clip)))]
        clips.append(np.stack(clip, axis=0))
    return np.stack(clips, axis=0)


def _tensor_scalar(value) -> float:
    return float(value.detach().reshape(-1).mean().cpu().item())


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Video evaluation requires torch because it reuses torch metrics."
        ) from exc
    return torch
