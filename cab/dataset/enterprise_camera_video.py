from __future__ import annotations

import fnmatch
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import VisionDataset


@dataclass(frozen=True)
class CameraSegment:
    """A timestamp-continuous run of frames from one camera."""

    session: str
    camera: str
    camera_dir: Path
    fps: float
    width: int
    height: int
    frame_paths: tuple[Path, ...]
    timestamps_ms: tuple[int, ...]


@dataclass(frozen=True)
class VideoClip:
    """A fixed-length view into a :class:`CameraSegment`."""

    segment_index: int
    start_index: int
    frame_count: int


class EnterpriseCameraVideoDataset(VisionDataset):
    """Read timestamped camera frames from enterprise collection sessions.

    The accepted layout is either ``root/ori_image/<camera>/<timestamp>.jpg``
    for one session, or ``root/<session>/ori_image/<camera>/<timestamp>.jpg``
    for multiple sessions. Each camera is treated as an independent video and
    is returned at its native resolution and native frame rate.

    ``front_wide`` cameras are expected to run at 20 Hz (50 ms interval); all
    other cameras are expected to run at 10 Hz (100 ms interval). A timestamp
    discontinuity starts a new segment, so a clip never crosses missing frames.
    Returned video tensors have shape ``(C, T, H, W)`` and values in ``[0, 1]``
    unless ``zero_mean=True`` is requested.
    """

    def __init__(
        self,
        root: str | Path,
        image_subdir: str = "ori_image",
        include_cameras: Sequence[str] | None = None,
        exclude_cameras: Sequence[str] | None = None,
        clip_len: int = 33,
        clip_stride: int = 32,
        max_clips_per_camera: int | None = None,
        zero_mean: bool = False,
        timestamp_tolerance_ms: int = 10,
        strict: bool = True,
    ) -> None:
        super().__init__(str(root))
        if clip_len <= 0:
            raise ValueError("clip_len must be greater than 0")
        if clip_stride <= 0:
            raise ValueError("clip_stride must be greater than 0")
        if max_clips_per_camera is not None and max_clips_per_camera <= 0:
            raise ValueError("max_clips_per_camera must be greater than 0 or None")
        if timestamp_tolerance_ms < 0:
            raise ValueError("timestamp_tolerance_ms must be non-negative")

        self.root_path = Path(root).expanduser()
        self.image_subdir = image_subdir
        self.include_cameras = self._normalise_patterns(include_cameras)
        self.exclude_cameras = self._normalise_patterns(exclude_cameras)
        self.clip_len = int(clip_len)
        self.clip_stride = int(clip_stride)
        self.max_clips_per_camera = max_clips_per_camera
        self.zero_mean = bool(zero_mean)
        self.timestamp_tolerance_ms = int(timestamp_tolerance_ms)
        self.strict = bool(strict)

        session_dirs = self._discover_session_dirs()
        self.segments = self._discover_segments(session_dirs)
        self.clips = self._build_clips()
        if not self.clips:
            raise ValueError(
                f"No valid {self.clip_len}-frame clips found under {self.root_path}. "
                "Check the root, camera filters, timestamps, and clip length."
            )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, object]:
        clip = self.clips[index]
        segment = self.segments[clip.segment_index]
        end = clip.start_index + clip.frame_count
        frame_paths = segment.frame_paths[clip.start_index:end]
        timestamps = segment.timestamps_ms[clip.start_index:end]

        frames = [self._read_frame(path, segment) for path in frame_paths]
        video = torch.stack(frames, dim=1)
        if self.zero_mean:
            video = video * 2.0 - 1.0

        start_timestamp = timestamps[0]
        end_timestamp = timestamps[-1]
        return {
            "img": video,
            "fpath": str(segment.camera_dir),
            "sample": (
                f"{segment.session}__{segment.camera}__"
                f"{start_timestamp}_{end_timestamp}"
            ),
            "session": segment.session,
            "camera": segment.camera,
            "fps": torch.tensor(segment.fps, dtype=torch.float32),
            "timestamps_ms": torch.tensor(timestamps, dtype=torch.int64),
            "original_size": torch.tensor(
                [segment.height, segment.width], dtype=torch.int64
            ),
        }

    @staticmethod
    def _normalise_patterns(patterns: Sequence[str] | None) -> tuple[str, ...]:
        if patterns is None:
            return ()
        if isinstance(patterns, str):
            return (patterns,)
        normalised = tuple(str(pattern) for pattern in patterns)
        if any(not pattern for pattern in normalised):
            raise ValueError("Camera patterns must not be empty")
        return normalised

    def _discover_session_dirs(self) -> list[Path]:
        root = self.root_path
        if not root.exists():
            raise FileNotFoundError(f"Enterprise video root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Enterprise video root is not a directory: {root}")

        if (root / self.image_subdir).is_dir():
            return [root]

        sessions = sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and (child / self.image_subdir).is_dir()
        )
        if not sessions:
            raise FileNotFoundError(
                f"No '{self.image_subdir}' directory found in {root} or its "
                "direct child session directories"
            )
        return sessions

    def _discover_segments(self, session_dirs: Iterable[Path]) -> list[CameraSegment]:
        segments: list[CameraSegment] = []
        selected_camera_count = 0

        for session_dir in session_dirs:
            image_root = session_dir / self.image_subdir
            camera_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
            for camera_dir in camera_dirs:
                camera = camera_dir.name
                if not self._camera_is_selected(camera):
                    continue
                selected_camera_count += 1
                frames = self._list_timestamped_frames(camera_dir)
                if not frames:
                    warnings.warn(
                        f"No valid timestamped JPG frames found in {camera_dir}",
                        stacklevel=2,
                    )
                    continue
                segments.extend(
                    self._split_camera_frames(
                        session=session_dir.name,
                        camera=camera,
                        camera_dir=camera_dir,
                        frames=frames,
                    )
                )

        if selected_camera_count == 0:
            raise ValueError(
                "No camera directories matched include_cameras/exclude_cameras "
                f"under {self.root_path}"
            )
        return segments

    def _camera_is_selected(self, camera: str) -> bool:
        if self.include_cameras and not any(
            fnmatch.fnmatchcase(camera, pattern) for pattern in self.include_cameras
        ):
            return False
        if any(fnmatch.fnmatchcase(camera, pattern) for pattern in self.exclude_cameras):
            return False
        return True

    def _list_timestamped_frames(self, camera_dir: Path) -> list[tuple[int, Path]]:
        timestamp_to_path: dict[int, Path] = {}
        for path in sorted(camera_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".jpg":
                continue
            if not path.stem.isdigit():
                self._handle_invalid(
                    f"JPG filename is not a numeric millisecond timestamp: {path}"
                )
                continue

            timestamp = int(path.stem)
            previous = timestamp_to_path.get(timestamp)
            if previous is not None:
                self._handle_invalid(
                    f"Duplicate timestamp {timestamp} in {camera_dir}: "
                    f"{previous.name} and {path.name}"
                )
                continue
            timestamp_to_path[timestamp] = path

        return sorted(timestamp_to_path.items())

    def _split_camera_frames(
        self,
        session: str,
        camera: str,
        camera_dir: Path,
        frames: list[tuple[int, Path]],
    ) -> list[CameraSegment]:
        fps, expected_delta_ms = self._camera_timing(camera)
        timestamps = [timestamp for timestamp, _ in frames]
        if len(timestamps) > 1:
            median_delta = float(np.median(np.diff(timestamps)))
            if abs(median_delta - expected_delta_ms) > self.timestamp_tolerance_ms:
                self._handle_invalid(
                    f"Unexpected frame interval in {camera_dir}: median "
                    f"{median_delta:g} ms, expected {expected_delta_ms} +/- "
                    f"{self.timestamp_tolerance_ms} ms for {fps:g} Hz"
                )

        runs: list[list[tuple[int, Path]]] = []
        current_run = [frames[0]]
        for previous, current in zip(frames, frames[1:]):
            delta_ms = current[0] - previous[0]
            if abs(delta_ms - expected_delta_ms) <= self.timestamp_tolerance_ms:
                current_run.append(current)
            else:
                runs.append(current_run)
                current_run = [current]
        runs.append(current_run)

        segments: list[CameraSegment] = []
        for run in runs:
            if len(run) < self.clip_len:
                warnings.warn(
                    f"Dropping short continuous segment in {camera_dir}: "
                    f"{len(run)} frames ({run[0][0]}..{run[-1][0]}), need "
                    f"at least {self.clip_len}",
                    stacklevel=2,
                )
                continue
            width, height = self._read_image_size(run[0][1])
            segments.append(
                CameraSegment(
                    session=session,
                    camera=camera,
                    camera_dir=camera_dir,
                    fps=fps,
                    width=width,
                    height=height,
                    frame_paths=tuple(path for _, path in run),
                    timestamps_ms=tuple(timestamp for timestamp, _ in run),
                )
            )
        return segments

    @staticmethod
    def _camera_timing(camera: str) -> tuple[float, int]:
        if camera.endswith("front_wide"):
            return 20.0, 50
        return 10.0, 100

    def _build_clips(self) -> list[VideoClip]:
        clips: list[VideoClip] = []
        camera_clip_counts: dict[tuple[str, str], int] = {}

        for segment_index, segment in enumerate(self.segments):
            camera_key = (segment.session, segment.camera)
            count = camera_clip_counts.get(camera_key, 0)
            for start in range(
                0,
                len(segment.frame_paths) - self.clip_len + 1,
                self.clip_stride,
            ):
                if (
                    self.max_clips_per_camera is not None
                    and count >= self.max_clips_per_camera
                ):
                    break
                clips.append(
                    VideoClip(
                        segment_index=segment_index,
                        start_index=start,
                        frame_count=self.clip_len,
                    )
                )
                count += 1
            camera_clip_counts[camera_key] = count
        return clips

    @staticmethod
    def _read_image_size(path: Path) -> tuple[int, int]:
        try:
            with Image.open(path) as image:
                return image.size
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Cannot read image header: {path}") from error

    @staticmethod
    def _read_frame(path: Path, segment: CameraSegment) -> torch.Tensor:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if image.size != (segment.width, segment.height):
                    raise ValueError(
                        f"Resolution changed within camera segment {segment.camera_dir}: "
                        f"expected {segment.width}x{segment.height}, found "
                        f"{image.width}x{image.height} at {path.name}"
                    )
                array = np.array(image, dtype=np.float32, copy=True) / 255.0
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Cannot decode JPG frame: {path}") from error
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _handle_invalid(self, message: str) -> None:
        if self.strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)


__all__ = ["CameraSegment", "EnterpriseCameraVideoDataset", "VideoClip"]
