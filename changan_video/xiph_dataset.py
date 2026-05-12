from __future__ import annotations

import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from .h264_writer import H264Writer, VideoWriteConfig, VideoWriteStats


@dataclass(frozen=True)
class XiphSample:
    url: str
    width: int
    height: int
    fps: Fraction
    frames: int | None = None


@dataclass(frozen=True)
class VideoSourceInfo:
    width: int
    height: int
    fps: Fraction
    frames: int | None = None
    duration_seconds: float | None = None


XIPH_SAMPLES: dict[str, XiphSample] = {
    "bus_qcif_15fps": XiphSample(
        url="https://media.xiph.org/video/derf/y4m/bus_qcif_15fps.y4m",
        width=176,
        height=144,
        fps=Fraction(15, 1),
        frames=75,
    ),
}


def _pick_video_rate(stream: av.video.stream.VideoStream) -> Fraction:
    rate = stream.average_rate or stream.base_rate or stream.guessed_rate
    if rate is None:
        warnings.warn(
            "Could not determine video frame rate; assuming 30 fps",
            RuntimeWarning,
            stacklevel=2,
        )
        return Fraction(30, 1)
    return Fraction(rate)


def probe_video_source(source: str | Path) -> VideoSourceInfo:
    """Open a video/Y4M source and return dimensions, fps, and frame count."""

    container = av.open(str(source), mode="r")
    try:
        stream = container.streams.video[0]
        fps = _pick_video_rate(stream)
        frames = stream.frames if stream.frames else None
        duration_seconds = None
        if frames is not None:
            duration_seconds = float(Fraction(frames, 1) / fps)

        return VideoSourceInfo(
            width=stream.width,
            height=stream.height,
            fps=fps,
            frames=frames,
            duration_seconds=duration_seconds,
        )
    finally:
        container.close()


def iter_video_frames(
    source: str | Path,
    limit: int | None = None,
    ndarray_format: str = "rgb24",
) -> Iterator[np.ndarray]:
    """Yield decoded frames from a video/Y4M source as uint8 numpy arrays."""

    container = av.open(str(source), mode="r")
    decoded = 0
    try:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format=ndarray_format)
            decoded += 1
            if limit is not None and decoded >= limit:
                break
    finally:
        container.close()


def transcode_video_source(
    source: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    codec: str = "libx264",
    crf: int | None = 23,
    preset: str | None = "veryfast",
) -> VideoWriteStats:
    """Decode a video/Y4M source and encode it as H.264."""

    container = av.open(str(source), mode="r")
    try:
        input_stream = container.streams.video[0]
        fps = _pick_video_rate(input_stream)

        config = VideoWriteConfig(
            output_path=output_path,
            width=input_stream.width,
            height=input_stream.height,
            fps=fps,
            codec=codec,
            crf=crf,
            preset=preset,
        )

        decoded = 0
        with H264Writer(config) as writer:
            for frame in container.decode(video=0):
                writer.write(frame)
                decoded += 1
                if limit is not None and decoded >= limit:
                    break
        return writer.stats
    finally:
        container.close()
