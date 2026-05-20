from __future__ import annotations

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


H264WriteConfig = VideoWriteConfig


class H264Writer(BaseVideoWriter):
    """Incremental H.264 writer."""


def encode_frames(
    frames: Iterable[FrameLike],
    config: VideoWriteConfig,
    progress: ProgressCallback | None = None,
) -> VideoWriteStats:
    """Write an iterable of numpy arrays or PyAV frames to H.264/MP4."""

    return encode_frames_with_writer(H264Writer, frames, config, progress)


def encode_records(
    records: Iterable[Any],
    config: VideoWriteConfig,
    frame_getter: FrameGetter,
    progress: ProgressCallback | None = None,
) -> VideoWriteStats:
    """Write H.264 frames pulled from database records."""

    return encode_records_with_writer(H264Writer, records, config, frame_getter, progress)


encode_frames_h264 = encode_frames
encode_records_h264 = encode_records
