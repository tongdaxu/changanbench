from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.datasets import VisionDataset


class SimpleVideoDataset(VisionDataset):
    """Read videos stored as frame folders.

    Each sample directory is expected to contain image frames. The returned
    tensor is in [0, 1] with shape (C, T, H, W), so DataLoader batches become
    (B, C, T, H, W).
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        clip_len: int = 32,
        image_size: int | None = None,
        zero_mean: bool = False,
        sampling: str = "uniform",
    ):
        super().__init__(root)
        if clip_len <= 0:
            raise ValueError("clip_len must be greater than 0")
        if sampling not in {"uniform", "first", "center"}:
            raise ValueError("sampling must be one of: uniform, first, center")

        self.root_path = Path(root)
        self.clip_len = int(clip_len)
        self.image_size = image_size
        self.zero_mean = bool(zero_mean)
        self.sampling = sampling
        self.samples = self._discover_samples(self.root_path)
        assert len(self.samples) > 0, "No frame folders found. Check the root."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_dir = self.samples[index]
        frame_paths = self._sample_frames(self._list_frames(sample_dir))
        frames = [self._read_frame(path) for path in frame_paths]
        video = torch.stack(frames, dim=1)
        if self.zero_mean:
            video = video * 2.0 - 1.0
        return {
            "img": video,
            "fpath": str(sample_dir),
        }

    def _discover_samples(self, root: Path) -> List[Path]:
        if not root.exists():
            raise FileNotFoundError(f"Video root does not exist: {root}")
        if self._list_frames(root):
            return [root]
        return sorted(
            path for path in root.rglob("*")
            if path.is_dir() and len(self._list_frames(path)) >= self.clip_len
        )

    def _list_frames(self, folder: Path) -> List[Path]:
        if not folder.exists() or not folder.is_dir():
            return []
        return sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMG_EXTS
        )

    def _sample_frames(self, frames: List[Path]) -> List[Path]:
        if len(frames) < self.clip_len:
            raise ValueError(f"Need {self.clip_len} frames, found {len(frames)}")
        if self.sampling == "first":
            return frames[: self.clip_len]
        if self.sampling == "center":
            start = max(0, (len(frames) - self.clip_len) // 2)
            return frames[start : start + self.clip_len]
        if self.clip_len == 1:
            return [frames[len(frames) // 2]]
        idxs = np.linspace(0, len(frames) - 1, self.clip_len)
        return [frames[int(round(i))] for i in idxs]

    def _read_frame(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.image_size is not None:
            img = self._resize_and_center_crop(img, int(self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    @staticmethod
    def _resize_and_center_crop(img: Image.Image, image_size: int) -> Image.Image:
        width, height = img.size
        short = min(width, height)
        scale = image_size / short
        new_width = max(image_size, int(round(width * scale)))
        new_height = max(image_size, int(round(height * scale)))
        img = img.resize((new_width, new_height), Image.BICUBIC)
        left = (new_width - image_size) // 2
        top = (new_height - image_size) // 2
        return img.crop((left, top, left + image_size, top + image_size))


class SequenceVideoDataset(VisionDataset):
    """Read sequence-style video datasets from frame folders or video files.

    This keeps structured datasets such as ScanNet and UCO3D behind the same
    CAB batch contract used by the video benchmark: ``{"img": (C,T,H,W)}``.
    """

    IMG_EXTS = SimpleVideoDataset.IMG_EXTS
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".y4m"}

    def __init__(
        self,
        root: str,
        source_type: str,
        clip_len: int = 32,
        image_size: int | None = None,
        zero_mean: bool = False,
        sampling: str = "uniform",
        frame_dir_pattern: str = "*",
        video_pattern: str = "*.mp4",
        scene_limit: int | None = None,
    ):
        super().__init__(root)
        if clip_len <= 0:
            raise ValueError("clip_len must be greater than 0")
        if sampling not in {"uniform", "first", "center"}:
            raise ValueError("sampling must be one of: uniform, first, center")
        if source_type not in {"frame_dirs", "videos"}:
            raise ValueError("source_type must be one of: frame_dirs, videos")

        self.root_path = Path(root)
        self.source_type = source_type
        self.clip_len = int(clip_len)
        self.image_size = image_size
        self.zero_mean = bool(zero_mean)
        self.sampling = sampling
        self.frame_dir_pattern = frame_dir_pattern
        self.video_pattern = video_pattern
        self.samples = self._discover_samples(self.root_path)
        if scene_limit is not None:
            self.samples = self.samples[: int(scene_limit)]
        assert len(self.samples) > 0, "No video sequences found. Check the root and pattern."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_path = self.samples[index]
        if self.source_type == "frame_dirs":
            video = self._read_frame_dir(sample_path)
        else:
            video = self._read_video_file(sample_path)
        if self.zero_mean:
            video = video * 2.0 - 1.0
        return {
            "img": video,
            "fpath": str(sample_path),
            "sample": sample_path.name,
        }

    def _discover_samples(self, root: Path) -> List[Path]:
        if not root.exists():
            raise FileNotFoundError(f"Video root does not exist: {root}")
        if self.source_type == "frame_dirs":
            return self._discover_frame_dirs(root)
        return self._discover_video_files(root)

    def _discover_frame_dirs(self, root: Path) -> List[Path]:
        if self._list_frames(root):
            return [root]
        return sorted(
            path for path in root.rglob(self.frame_dir_pattern)
            if path.is_dir() and len(self._list_frames(path)) >= self.clip_len
        )

    def _discover_video_files(self, root: Path) -> List[Path]:
        if root.is_file() and root.suffix.lower() in self.VIDEO_EXTS:
            return [root]
        return sorted(
            path for path in root.rglob(self.video_pattern)
            if path.is_file() and path.suffix.lower() in self.VIDEO_EXTS
        )

    def _read_frame_dir(self, sample_dir: Path) -> torch.Tensor:
        frame_paths = self._sample_items(self._list_frames(sample_dir))
        frames = [self._read_frame(path) for path in frame_paths]
        return torch.stack(frames, dim=1)

    def _read_video_file(self, video_path: Path) -> torch.Tensor:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise ValueError(f"Could not determine frame count for {video_path}")
        indices = self._sample_indices(frame_count)
        selected_indices = set(indices)
        selected_frames = {}
        last_index = max(indices)
        try:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index in selected_indices:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
                    if self.image_size is not None:
                        tensor = self._resize_and_center_crop_tensor(tensor)
                    selected_frames[frame_index] = tensor
                if frame_index >= last_index:
                    break
                frame_index += 1
        finally:
            cap.release()

        missing = [index for index in indices if index not in selected_frames]
        if missing:
            raise ValueError(
                f"Could not decode {len(missing)} sampled frames from {video_path}; "
                f"first missing index: {missing[0]}"
            )
        frames = [selected_frames[index] for index in indices]
        return torch.stack(frames, dim=1).contiguous()

    def _list_frames(self, folder: Path) -> List[Path]:
        if not folder.exists() or not folder.is_dir():
            return []
        return sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMG_EXTS
        )

    def _sample_items(self, items: List[Path]) -> List[Path]:
        if len(items) < self.clip_len:
            raise ValueError(f"Need {self.clip_len} frames, found {len(items)}")
        return [items[idx] for idx in self._sample_indices(len(items))]

    def _sample_indices(self, count: int) -> List[int]:
        if count < self.clip_len:
            raise ValueError(f"Need {self.clip_len} items, found {count}")
        if self.sampling == "first":
            return list(range(self.clip_len))
        if self.sampling == "center":
            start = max(0, (count - self.clip_len) // 2)
            return list(range(start, start + self.clip_len))
        if self.clip_len == 1:
            return [count // 2]
        idxs = np.linspace(0, count - 1, self.clip_len)
        return [int(round(i)) for i in idxs]

    def _read_frame(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.image_size is not None:
            img = SimpleVideoDataset._resize_and_center_crop(img, int(self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _resize_and_center_crop_tensor(self, frame: torch.Tensor) -> torch.Tensor:
        image_size = int(self.image_size)
        _, height, width = frame.shape
        short = min(width, height)
        scale = image_size / short
        new_width = max(image_size, int(round(width * scale)))
        new_height = max(image_size, int(round(height * scale)))
        frame = F.resize(frame, [new_height, new_width], antialias=True)
        return F.center_crop(frame, [image_size, image_size])
