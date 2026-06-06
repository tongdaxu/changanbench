from __future__ import annotations

import csv
import json
from pathlib import Path

from cab.evaluations.video_types import VideoEvaluationResult


def write_frame_metrics_csv(result: VideoEvaluationResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_metric_names = [
        name
        for name in result.metric_names
        if result.frame_metrics and name in result.frame_metrics[0].values
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", *frame_metric_names])
        writer.writeheader()
        for item in result.frame_metrics:
            writer.writerow({"frame": item.frame_index, **item.values})


def write_summary_json(
    result: VideoEvaluationResult,
    output_path: str | Path,
    *,
    include_frames: bool = False,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(include_frames=include_frames), handle, indent=2)
        handle.write("\n")
