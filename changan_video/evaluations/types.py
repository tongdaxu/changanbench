from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameMetricResult:
    frame_index: int
    values: dict[str, float]


@dataclass(frozen=True)
class MetricSummary:
    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }


@dataclass(frozen=True)
class VideoEvaluationResult:
    reference: str
    distorted: str
    width: int
    height: int
    frames: int
    metric_names: tuple[str, ...]
    metrics: dict[str, MetricSummary]
    frame_metrics: list[FrameMetricResult]
    distorted_bytes: int | None = None
    bits_per_pixel: float | None = None

    def to_dict(self, include_frames: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "reference": self.reference,
            "distorted": self.distorted,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "metrics": {
                name: self.metrics[name].to_dict() for name in self.metric_names
            },
            "distorted_bytes": self.distorted_bytes,
            "bits_per_pixel": self.bits_per_pixel,
        }
        if include_frames:
            data["frames_detail"] = [
                {"frame": item.frame_index, **item.values}
                for item in self.frame_metrics
            ]
        return data
