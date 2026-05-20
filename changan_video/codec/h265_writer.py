from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from .video_writer import (
    BaseVideoWriter,
    FrameGetter,
    FrameLike,
    ProgressCallback,
    VideoWriteConfig,
    VideoWriteStats,
    encode_frames_with_writer,
    encode_records_with_writer,
)


@dataclass(frozen=True)
class H265WriteConfig(VideoWriteConfig):
    """Configuration for writing H.265/HEVC with PyAV."""

    codec: str = "libx265"
    crf: int | None = 28


class H265Writer(BaseVideoWriter):
    """Incremental H.265/HEVC writer."""

    def __init__(self, config: VideoWriteConfig):
        if config.codec == "libx264":
            config = replace(config, codec="libx265")
        super().__init__(config)


def encode_frames_h265(
    frames: Iterable[FrameLike],
    config: VideoWriteConfig,
    progress: ProgressCallback | None = None,
) -> VideoWriteStats:
    """Write an iterable of numpy arrays or PyAV frames to H.265/HEVC."""

    return encode_frames_with_writer(H265Writer, frames, config, progress)


def encode_records_h265(
    records: Iterable[Any],
    config: VideoWriteConfig,
    frame_getter: FrameGetter,
    progress: ProgressCallback | None = None,
) -> VideoWriteStats:
    """Write H.265/HEVC frames pulled from database records."""

    return encode_records_with_writer(H265Writer, records, config, frame_getter, progress)
