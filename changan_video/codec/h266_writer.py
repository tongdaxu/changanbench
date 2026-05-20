from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import av

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
class H266WriteConfig(VideoWriteConfig):
    """Configuration for writing H.266/VVC with PyAV."""

    codec: str = "libvvenc"
    crf: int | None = None
    qp: int | None = 32
    preset: str | None = "medium"


class H266Writer(BaseVideoWriter):
    """Incremental H.266/VVC writer.

    Requires a PyAV/FFmpeg build with an H.266/VVC encoder, typically libvvenc.
    """

    def __init__(self, config: VideoWriteConfig):
        if config.codec == "libx264":
            preset = "medium" if config.preset == "veryfast" else config.preset
            config = replace(config, codec="libvvenc", crf=None, qp=32, preset=preset)
        _ensure_h266_encoder(config.codec)
        super().__init__(config)


def encode_frames_h266(
    frames: Iterable[FrameLike],
    config: VideoWriteConfig,
    progress: ProgressCallback | None = None,
) -> VideoWriteStats:
    """Write an iterable of numpy arrays or PyAV frames to H.266/VVC."""

    return encode_frames_with_writer(H266Writer, frames, config, progress)


def encode_records_h266(
    records: Iterable[Any],
    config: VideoWriteConfig,
    frame_getter: FrameGetter,
    progress: ProgressCallback | None = None,
) -> VideoWriteStats:
    """Write H.266/VVC frames pulled from database records."""

    return encode_records_with_writer(H266Writer, records, config, frame_getter, progress)


def _ensure_h266_encoder(codec: str) -> None:
    try:
        av.codec.Codec(codec, "w")
    except Exception as exc:
        raise RuntimeError(
            "H.266/VVC encoder "
            f"{codec!r} is not available in this PyAV/FFmpeg build. "
            "Install a build with libvvenc support, or pass another available "
            "H.266 encoder name in H266WriteConfig(codec=...)."
        ) from exc
