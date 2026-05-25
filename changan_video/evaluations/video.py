from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from changan_video.evaluations.config import (
    DEFAULT_METRICS,
    FRAME_METRICS,
    VIDEO_METRICS,
    check_dependencies,
    fvd_options_from_config,
    metrics_from_config,
    normalize_metrics,
    progress_callback,
    str_to_bool,
    vggt_metric_from_config,
    video_path_from_config,
)
from changan_video.evaluations.io import write_frame_metrics_csv, write_summary_json
from changan_video.evaluations.types import FrameMetricResult, VideoEvaluationResult
from changan_video.evaluations.runners import (
    FidFeatureRunner,
    VideoFrameMetricRunner,
    fid_summary,
    fvd_summary,
    resolve_torch_device,
    scalar_summary,
    summarize_frame_metrics,
)

__all__ = [
    "DEFAULT_METRICS",
    "check_dependencies",
    "evaluate_video_pair",
    "fvd_options_from_config",
    "metrics_from_config",
    "normalize_metrics",
    "progress_callback",
    "str_to_bool",
    "vggt_metric_from_config",
    "video_path_from_config",
    "write_frame_metrics_csv",
    "write_summary_json",
]


def evaluate_video_pair(
    reference: str | Path,
    distorted: str | Path,
    metrics: Sequence[str] | None = None,
    *,
    device: str = "auto",
    limit: int | None = None,
    zero_mean: bool = False,
    lpips_network: str = "alex",
    fvd_clip_length: int = 16,
    fvd_clip_stride: int = 16,
    fvd_model_path: str | None = None,
    vggt_metric=None,
    resize_distorted: bool = False,
    allow_frame_count_mismatch: bool = False,
    progress: Callable[[int], None] | None = None,
) -> VideoEvaluationResult:
    import av

    metric_names = normalize_metrics(metrics or DEFAULT_METRICS)
    frame_metric_names = tuple(name for name in metric_names if name in FRAME_METRICS)
    video_metric_names = tuple(name for name in metric_names if name in VIDEO_METRICS)

    frame_runner = (
        VideoFrameMetricRunner(
            metric_names=frame_metric_names,
            device=device,
            zero_mean=zero_mean,
            lpips_network=lpips_network,
        )
        if frame_metric_names
        else None
    )
    fid_runner = FidFeatureRunner(device=device) if "fid" in video_metric_names else None

    state = _EvaluationState(reference=str(reference), distorted=str(distorted))
    reference_container = av.open(state.reference, mode="r")
    distorted_container = av.open(state.distorted, mode="r")
    try:
        reference_iter = iter(reference_container.decode(video=0))
        distorted_iter = iter(distorted_container.decode(video=0))
        frame_index = 0
        while limit is None or frame_index < limit:
            reference_frame = _next_frame(reference_iter)
            distorted_frame = _next_frame(distorted_iter)
            if reference_frame is None and distorted_frame is None:
                break
            if reference_frame is None or distorted_frame is None:
                if allow_frame_count_mismatch:
                    break
                raise ValueError(
                    "Frame count mismatch at frame "
                    f"{frame_index}: reference ended={reference_frame is None}, "
                    f"distorted ended={distorted_frame is None}"
                )

            reference_array, distorted_array = _frame_arrays(
                reference_frame,
                distorted_frame,
                frame_index=frame_index,
                resize_distorted=resize_distorted,
            )
            _record_frame(
                state,
                frame_index,
                reference_array,
                distorted_array,
                frame_runner=frame_runner,
                fid_runner=fid_runner,
                keep_video_frames=bool({"fvd", "vggt"} & set(video_metric_names)),
            )

            frame_index += 1
            if progress is not None:
                progress(frame_index)
    finally:
        reference_container.close()
        distorted_container.close()

    if not state.frame_metrics or state.width is None or state.height is None:
        raise ValueError("No frames were evaluated")

    summaries = summarize_frame_metrics(state.frame_metrics, frame_metric_names)
    if "fid" in video_metric_names:
        summaries["fid"] = fid_summary(
            state.reference_fid_features,
            state.distorted_fid_features,
        )
    if "fvd" in video_metric_names:
        summaries["fvd"] = fvd_summary(
            state.reference_video_frames,
            state.distorted_video_frames,
            device=resolve_torch_device(device),
            clip_length=fvd_clip_length,
            clip_stride=fvd_clip_stride,
            model_path=fvd_model_path,
        )
    if "vggt" in video_metric_names:
        if vggt_metric is None:
            raise ValueError("Metric 'vggt' requires a configured VGGTVideoMetric instance")
        vggt_values = vggt_metric(state.reference_video_frames, state.distorted_video_frames)
        for name, value in vggt_values.items():
            summaries[name] = scalar_summary(name, value, count=len(state.frame_metrics))

    return VideoEvaluationResult(
        reference=state.reference,
        distorted=state.distorted,
        width=state.width,
        height=state.height,
        frames=len(state.frame_metrics),
        metric_names=tuple(summaries.keys()),
        metrics=summaries,
        frame_metrics=state.frame_metrics,
        distorted_bytes=_local_file_size(distorted),
        bits_per_pixel=_bits_per_pixel(distorted, len(state.frame_metrics), state.width, state.height),
    )


class _EvaluationState:
    def __init__(self, *, reference: str, distorted: str) -> None:
        self.reference = reference
        self.distorted = distorted
        self.frame_metrics: list[FrameMetricResult] = []
        self.reference_fid_features = []
        self.distorted_fid_features = []
        self.reference_video_frames = []
        self.distorted_video_frames = []
        self.width: int | None = None
        self.height: int | None = None


def _record_frame(
    state: _EvaluationState,
    frame_index: int,
    reference_array,
    distorted_array,
    *,
    frame_runner: VideoFrameMetricRunner | None,
    fid_runner: FidFeatureRunner | None,
    keep_video_frames: bool,
) -> None:
    if state.width is None or state.height is None:
        state.height, state.width = reference_array.shape[:2]

    values = frame_runner.score(reference_array, distorted_array) if frame_runner else {}
    state.frame_metrics.append(FrameMetricResult(frame_index, values))
    if fid_runner is not None:
        state.reference_fid_features.append(fid_runner.extract(reference_array))
        state.distorted_fid_features.append(fid_runner.extract(distorted_array))
    if keep_video_frames:
        state.reference_video_frames.append(reference_array)
        state.distorted_video_frames.append(distorted_array)


def _frame_arrays(reference_frame, distorted_frame, *, frame_index: int, resize_distorted: bool):
    reference_array = reference_frame.to_ndarray(format="rgb24")
    distorted_array = distorted_frame.to_ndarray(format="rgb24")

    if reference_array.shape[:2] == distorted_array.shape[:2]:
        return reference_array, distorted_array
    if not resize_distorted:
        raise ValueError(
            "Frame size mismatch at frame "
            f"{frame_index}: reference {reference_array.shape[1]}x{reference_array.shape[0]}, "
            f"distorted {distorted_array.shape[1]}x{distorted_array.shape[0]}. "
            "Set resize_distorted=True to resize."
        )

    distorted_frame = distorted_frame.reformat(
        width=reference_array.shape[1],
        height=reference_array.shape[0],
        format="rgb24",
    )
    return reference_array, distorted_frame.to_ndarray(format="rgb24")


def _bits_per_pixel(path: str | Path, frames: int, width: int, height: int) -> float | None:
    size = _local_file_size(path)
    if size is None:
        return None
    return size * 8.0 / (frames * width * height)


def _local_file_size(path: str | Path) -> int | None:
    try:
        candidate = Path(path)
    except TypeError:
        return None
    if candidate.exists() and candidate.is_file():
        return candidate.stat().st_size
    return None


def _next_frame(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None
