from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar, Union

import av
import numpy as np


FrameLike = Union[np.ndarray, av.VideoFrame]
FpsLike = Union[int, float, Fraction]
FrameGetter = Callable[[Any], FrameLike]
ProgressCallback = Callable[[int], None]


@dataclass(frozen=True)
class VideoWriteConfig:
    """Configuration for writing encoded video with PyAV."""

    output_path: str | Path
    width: int
    height: int
    fps: FpsLike = 30
    codec: str = "libx264"
    container_format: str | None = None
    input_format: str = "rgb24"
    pix_fmt: str = "yuv420p"
    crf: int | None = 23
    qp: int | None = None
    preset: str | None = "veryfast"
    tune: str | None = None
    profile: str | None = None
    bit_rate: int | None = None
    gop_size: int | None = None
    max_b_frames: int | None = None
    strict_size: bool = True
    codec_options: dict[str, str] | None = None


@dataclass(frozen=True)
class VideoWriteStats:
    output_path: str
    frames: int
    fps: Fraction
    duration_seconds: float
    bytes_written: int | None


def _as_fraction_fps(fps: FpsLike) -> Fraction:
    if isinstance(fps, Fraction):
        value = fps
    elif isinstance(fps, int):
        value = Fraction(fps, 1)
    elif isinstance(fps, float):
        value = Fraction(fps).limit_denominator(100_000)
    else:
        raise TypeError(f"Unsupported fps type: {type(fps)!r}")

    if value <= 0:
        raise ValueError("fps must be greater than 0")
    return value


def infer_container_format(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".h264", ".264"}:
        return "h264"
    if suffix in {".h265", ".265", ".hevc"}:
        return "hevc"
    if suffix in {".h266", ".266", ".vvc"}:
        return "vvc"
    if suffix == ".mp4":
        return "mp4"
    return None


class BaseVideoWriter:
    """Incremental PyAV video writer shared by codec-specific wrappers."""

    def __init__(self, config: VideoWriteConfig):
        self.config = config
        self.output_path = Path(config.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.fps = _as_fraction_fps(config.fps)
        self.time_base = Fraction(self.fps.denominator, self.fps.numerator)
        container_format = config.container_format or infer_container_format(self.output_path)

        self.container = av.open(str(self.output_path), mode="w", format=container_format)
        self.stream = self.container.add_stream(config.codec, rate=self.fps)
        self.stream.width = config.width
        self.stream.height = config.height
        self.stream.pix_fmt = config.pix_fmt
        self.stream.codec_context.time_base = self.time_base

        if config.bit_rate is not None:
            self.stream.codec_context.bit_rate = int(config.bit_rate)
        if config.gop_size is not None:
            self.stream.codec_context.gop_size = int(config.gop_size)
        if config.max_b_frames is not None:
            self.stream.codec_context.max_b_frames = int(config.max_b_frames)

        options = self._codec_options(config)
        if options:
            self.stream.options = options

        self._frames = 0
        self._closed = False

    def __enter__(self) -> BaseVideoWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def frames_written(self) -> int:
        return self._frames

    @property
    def stats(self) -> VideoWriteStats:
        return self._stats()

    def write(self, frame: FrameLike) -> int:
        """Encode one frame and mux any packets that become available."""

        if self._closed:
            raise RuntimeError(f"Cannot write to a closed {type(self).__name__}")

        video_frame = self._to_video_frame(frame)
        video_frame.pts = self._frames
        video_frame.time_base = self.time_base

        packets = 0
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)
            packets += 1

        self._frames += 1
        return packets

    def close(self) -> VideoWriteStats:
        """Flush delayed packets, close the container, and return stats."""

        if self._closed:
            return self._stats()

        for packet in self.stream.encode():
            self.container.mux(packet)

        self.container.close()
        self._closed = True
        return self._stats()

    def _stats(self) -> VideoWriteStats:
        size = None
        if self.output_path.exists():
            size = self.output_path.stat().st_size

        return VideoWriteStats(
            output_path=str(self.output_path),
            frames=self._frames,
            fps=self.fps,
            duration_seconds=float(Fraction(self._frames, 1) / self.fps),
            bytes_written=size,
        )

    def _codec_options(self, config: VideoWriteConfig) -> dict[str, str]:
        options: dict[str, str] = {}
        if config.qp is not None:
            options["qp"] = str(config.qp)
        elif config.crf is not None and config.bit_rate is None:
            options["crf"] = str(config.crf)
        if config.preset:
            options["preset"] = config.preset
        if config.tune:
            options["tune"] = config.tune
        if config.profile:
            options["profile"] = config.profile
        if config.codec_options:
            options.update(config.codec_options)
        return options

    def _to_video_frame(self, frame: FrameLike) -> av.VideoFrame:
        if isinstance(frame, av.VideoFrame):
            size_mismatch = (
                frame.width != self.config.width or frame.height != self.config.height
            )
            if self.config.strict_size and size_mismatch:
                raise ValueError(
                    "Frame size mismatch: "
                    f"expected {self.config.width}x{self.config.height}, "
                    f"got {frame.width}x{frame.height}"
                )
            # Always reformat to get an owned copy; avoids mutating the caller's frame.
            return frame.reformat(
                width=self.config.width,
                height=self.config.height,
                format=self.config.pix_fmt,
            )

        array = self._normalize_array(frame)
        video_frame = av.VideoFrame.from_ndarray(array, format=self.config.input_format)
        size_mismatch = (
            video_frame.width != self.config.width or video_frame.height != self.config.height
        )
        if self.config.strict_size and size_mismatch:
            raise ValueError(
                "Frame size mismatch: "
                f"expected {self.config.width}x{self.config.height}, "
                f"got {video_frame.width}x{video_frame.height}"
            )
        if size_mismatch or video_frame.format.name != self.config.pix_fmt:
            return video_frame.reformat(
                width=self.config.width,
                height=self.config.height,
                format=self.config.pix_fmt,
            )
        return video_frame

    def _normalize_array(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be numpy.ndarray, got {type(frame)!r}")

        array = frame
        if array.dtype != np.uint8:
            raise TypeError(f"frame dtype must be uint8, got {array.dtype}")

        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim != 3:
            raise ValueError(f"frame must be HxW or HxWxC, got shape {array.shape}")

        if array.shape[2] not in (3, 4):
            raise ValueError(f"frame channel count must be 3 or 4, got shape {array.shape}")

        return np.ascontiguousarray(array)


WriterT = TypeVar("WriterT", bound=BaseVideoWriter)


def encode_frames_with_writer(
    writer_type: type[WriterT],
    frames: Iterable[FrameLike],
    config: VideoWriteConfig,
    progress: ProgressCallback | None = None,
) -> VideoWriteStats:
    with writer_type(config) as writer:
        for frame in frames:
            writer.write(frame)
            if progress:
                progress(writer.frames_written)
    return writer.stats


def encode_records_with_writer(
    writer_type: type[WriterT],
    records: Iterable[Any],
    config: VideoWriteConfig,
    frame_getter: FrameGetter,
    progress: ProgressCallback | None = None,
) -> VideoWriteStats:
    with writer_type(config) as writer:
        for record in records:
            writer.write(frame_getter(record))
            if progress:
                progress(writer.frames_written)
    return writer.stats
