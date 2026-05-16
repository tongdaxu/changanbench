from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
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
