from __future__ import annotations

import urllib.request
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class HevcCtcSequence:
    name: str
    class_name: str
    width: int
    height: int
    fps: Fraction
    frames: int
    filename: str
    aliases: tuple[str, ...] = ()
    url: str | None = None


def _seq(
    name: str,
    class_name: str,
    width: int,
    height: int,
    fps: int,
    frames: int,
    filename: str | None = None,
    aliases: tuple[str, ...] = (),
    url: str | None = None,
) -> HevcCtcSequence:
    return HevcCtcSequence(
        name=name,
        class_name=class_name,
        width=width,
        height=height,
        fps=Fraction(fps, 1),
        frames=frames,
        filename=filename or f"{name}_{width}x{height}_{fps}.yuv",
        aliases=aliases,
        url=url,
    )


_XIPH_Y4M = "https://media.xiph.org/video/derf/y4m"


HEVC_CTC_SEQUENCES: dict[str, HevcCtcSequence] = {
    # Class A
    "Traffic": _seq("Traffic", "ClassA", 2560, 1600, 30, 150, "Traffic_2560x1600_30_crop.yuv"),
    "PeopleOnStreet": _seq(
        "PeopleOnStreet",
        "ClassA",
        2560,
        1600,
        30,
        150,
        "PeopleOnStreet_2560x1600_30_crop.yuv",
    ),
    # Class B
    "Kimono1": _seq("Kimono1", "ClassB", 1920, 1080, 24, 240),
    "ParkScene": _seq("ParkScene", "ClassB", 1920, 1080, 24, 240),
    "Cactus": _seq("Cactus", "ClassB", 1920, 1080, 50, 500),
    "BasketballDrive": _seq("BasketballDrive", "ClassB", 1920, 1080, 50, 500),
    "BQTerrace": _seq("BQTerrace", "ClassB", 1920, 1080, 60, 600),
    # Class C
    "BasketballDrill": _seq("BasketballDrill", "ClassC", 832, 480, 50, 500),
    "BQMall": _seq("BQMall", "ClassC", 832, 480, 60, 600),
    "PartyScene": _seq("PartyScene", "ClassC", 832, 480, 50, 500),
    "RaceHorsesC": _seq("RaceHorses", "ClassC", 832, 480, 30, 300),
    # Class D
    "BasketballPass": _seq("BasketballPass", "ClassD", 416, 240, 50, 500),
    "BQSquare": _seq("BQSquare", "ClassD", 416, 240, 60, 600),
    "BlowingBubbles": _seq("BlowingBubbles", "ClassD", 416, 240, 50, 500),
    "RaceHorsesD": _seq("RaceHorses", "ClassD", 416, 240, 30, 300),
    # Class E. Xiph hosts y4m copies for these three.
    "FourPeople": _seq(
        "FourPeople",
        "ClassE",
        1280,
        720,
        60,
        600,
        aliases=("FourPeople_1280x720_60.y4m",),
        url=f"{_XIPH_Y4M}/FourPeople_1280x720_60.y4m",
    ),
    "Johnny": _seq(
        "Johnny",
        "ClassE",
        1280,
        720,
        60,
        600,
        aliases=("Johnny_1280x720_60.y4m",),
        url=f"{_XIPH_Y4M}/Johnny_1280x720_60.y4m",
    ),
    "KristenAndSara": _seq(
        "KristenAndSara",
        "ClassE",
        1280,
        720,
        60,
        600,
        aliases=("KristenAndSara_1280x720_60.y4m",),
        url=f"{_XIPH_Y4M}/KristenAndSara_1280x720_60.y4m",
    ),
    # Class F screen-content sequences.
    "BasketballDrillText": _seq("BasketballDrillText", "ClassF", 832, 480, 50, 500),
    "ChinaSpeed": _seq("ChinaSpeed", "ClassF", 1024, 768, 30, 500),
    "SlideEditing": _seq("SlideEditing", "ClassF", 1280, 720, 30, 300),
    "SlideShow": _seq("SlideShow", "ClassF", 1280, 720, 20, 500),
}


class HevcCtcVideoDataset(torch.utils.data.Dataset):
    """HEVC common-test-condition video sequences.

    The standard CTC clips are usually distributed as raw 8-bit YUV420 files.
    This dataset can also read y4m/video files when they are present.
    """

    VIDEO_EXTS = {".y4m", ".mp4", ".mov", ".mkv", ".avi", ".webm"}

    def __init__(
        self,
        root: str | Path = "./cache/hevc_ctc",
        sequences: Sequence[str | dict] | str | None = None,
        classes: Sequence[str] | str | None = ("ClassB", "ClassC", "ClassD", "ClassE"),
        clip_len: int = 32,
        image_size: int | None = None,
        zero_mean: bool = False,
        sampling: str = "uniform",
        download: bool = False,
        overwrite: bool = False,
        frame_limit: int | None = None,
        color_matrix: str = "bt709",
    ) -> None:
        if clip_len <= 0:
            raise ValueError("clip_len must be greater than 0")
        if sampling not in {"uniform", "first", "center"}:
            raise ValueError("sampling must be one of: uniform, first, center")
        if color_matrix not in {"bt601", "bt709"}:
            raise ValueError("color_matrix must be one of: bt601, bt709")

        self.root = Path(root)
        self.clip_len = int(clip_len)
        self.image_size = image_size
        self.zero_mean = bool(zero_mean)
        self.sampling = sampling
        self.download = bool(download)
        self.overwrite = bool(overwrite)
        self.frame_limit = frame_limit
        self.color_matrix = color_matrix
        self.items = self._build_items(sequences, classes)
        if not self.items:
            raise ValueError("No HEVC CTC sequences selected.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        sequence, path = self.items[index]
        if path.suffix.lower() in self.VIDEO_EXTS:
            frames = self._read_container_frames(path, sequence)
        else:
            frames = self._read_yuv420p_frames(path, sequence)
        tensors = [self._frame_to_tensor(frame) for frame in frames]
        video = torch.stack(tensors, dim=1).contiguous()
        if self.zero_mean:
            video = video * 2.0 - 1.0
        return {
            "img": video,
            "fpath": str(path),
            "sample": sequence.name,
            "class_name": sequence.class_name,
        }

    def _build_items(
        self,
        sequences: Sequence[str | dict] | str | None,
        classes: Sequence[str] | str | None,
    ) -> list[tuple[HevcCtcSequence, Path]]:
        if sequences is None:
            class_names = self._normalize_string_list(classes)
            selected = [
                seq
                for seq in HEVC_CTC_SEQUENCES.values()
                if not class_names or seq.class_name in class_names
            ]
        else:
            selected = [self._resolve_sequence_config(seq) for seq in self._normalize_sequence_list(sequences)]

        items = []
        for sequence in selected:
            path = self._resolve_path(sequence)
            items.append((sequence, path))
        return items

    def _resolve_sequence_config(self, entry: str | dict) -> HevcCtcSequence:
        if isinstance(entry, str):
            if entry not in HEVC_CTC_SEQUENCES:
                known = ", ".join(sorted(HEVC_CTC_SEQUENCES))
                raise ValueError(f"Unknown HEVC CTC sequence {entry!r}. Known sequences: {known}")
            return HEVC_CTC_SEQUENCES[entry]

        name = str(entry["name"])
        if "base" in entry:
            if entry["base"] not in HEVC_CTC_SEQUENCES:
                raise ValueError(f"Unknown base HEVC CTC sequence {entry['base']!r}")
            base = HEVC_CTC_SEQUENCES[entry["base"]]
        else:
            base = HevcCtcSequence(
                name=name,
                class_name=str(entry.get("class_name", "Custom")),
                width=int(entry["width"]),
                height=int(entry["height"]),
                fps=Fraction(int(entry.get("fps", 30)), 1),
                frames=int(entry["frames"]),
                filename=str(entry.get("filename", f"{name}.yuv")),
            )
        updates = {k: v for k, v in entry.items() if hasattr(base, k) and k != "fps"}
        if "fps" in entry:
            updates["fps"] = Fraction(int(entry["fps"]), 1)
        return replace(base, **updates)

    def _resolve_path(self, sequence: HevcCtcSequence) -> Path:
        candidates = self._candidate_paths(sequence)
        for path in candidates:
            if path.exists():
                return path

        if self.download:
            if sequence.url is None:
                raise FileNotFoundError(
                    f"{sequence.name} is not present under {self.root} and no download URL is configured."
                )
            return self._download(sequence)

        candidate_text = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"HEVC CTC sequence {sequence.name} not found. Checked:\n  {candidate_text}"
        )

    def _candidate_paths(self, sequence: HevcCtcSequence) -> list[Path]:
        names = (sequence.filename, *sequence.aliases)
        prefixes = (
            self.root,
            self.root / sequence.class_name,
            self.root / sequence.name,
            self.root / sequence.class_name.lower(),
            self.root / sequence.name.lower(),
        )
        return [prefix / name for prefix in prefixes for name in names]

    def _download(self, sequence: HevcCtcSequence) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        filename = sequence.url.rsplit("/", 1)[-1]
        dest = self.root / filename
        if dest.exists() and not self.overwrite:
            return dest
        print(f"Downloading {sequence.name} from {sequence.url} ...", flush=True)
        urllib.request.urlretrieve(sequence.url, dest)
        return dest

    def _read_container_frames(self, path: Path, sequence: HevcCtcSequence) -> list[np.ndarray]:
        import av

        source_frame_count = sequence.frames
        if self.frame_limit is not None:
            source_frame_count = min(source_frame_count, int(self.frame_limit))
        indices = set(self._sample_indices(source_frame_count))
        last_index = max(indices)

        container = av.open(str(path), mode="r")
        frames = []
        try:
            for idx, frame in enumerate(container.decode(video=0)):
                if idx in indices:
                    frames.append(frame.to_ndarray(format="rgb24"))
                if idx >= last_index:
                    break
        finally:
            container.close()
        if len(frames) < self.clip_len:
            raise ValueError(f"Need {self.clip_len} frames, found {len(frames)} in {path}")
        return frames

    def _read_yuv420p_frames(self, path: Path, sequence: HevcCtcSequence) -> list[np.ndarray]:
        width, height = sequence.width, sequence.height
        y_size = width * height
        uv_size = (width // 2) * (height // 2)
        frame_size = y_size + uv_size * 2
        max_frames = sequence.frames
        if self.frame_limit is not None:
            max_frames = min(max_frames, int(self.frame_limit))
        indices = self._sample_indices(max_frames)

        frames = []
        with path.open("rb") as fh:
            for idx in indices:
                fh.seek(frame_size * idx)
                raw = fh.read(frame_size)
                if len(raw) < frame_size:
                    break
                frames.append(self._yuv420p_to_rgb(raw, width, height, y_size, uv_size))
        if len(frames) < self.clip_len:
            raise ValueError(f"Need {self.clip_len} frames, found {len(frames)} in {path}")
        return frames

    def _yuv420p_to_rgb(
        self,
        raw: bytes,
        width: int,
        height: int,
        y_size: int,
        uv_size: int,
    ) -> np.ndarray:
        frame = np.frombuffer(raw, dtype=np.uint8)
        y = frame[:y_size].reshape(height, width).astype(np.float32)
        u = frame[y_size : y_size + uv_size].reshape(height // 2, width // 2).astype(np.float32)
        v = frame[y_size + uv_size : y_size + uv_size * 2].reshape(height // 2, width // 2).astype(np.float32)
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1) - 128.0
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1) - 128.0
        if self.color_matrix == "bt709":
            r = y + 1.5748 * v
            g = y - 0.1873 * u - 0.4681 * v
            b = y + 1.8556 * u
        else:
            r = y + 1.4020 * v
            g = y - 0.3441 * u - 0.7141 * v
            b = y + 1.7720 * u
        return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

    def _sample_indices(self, count: int) -> list[int]:
        if count < self.clip_len:
            raise ValueError(f"Need {self.clip_len} frames, found {count}")
        if self.sampling == "first":
            return list(range(self.clip_len))
        if self.sampling == "center":
            start = max(0, (count - self.clip_len) // 2)
            return list(range(start, start + self.clip_len))
        if self.clip_len == 1:
            return [count // 2]
        idxs = np.linspace(0, count - 1, self.clip_len)
        return [int(round(i)) for i in idxs]

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        if self.image_size is not None:
            image = Image.fromarray(frame, mode="RGB")
            frame = np.asarray(self._resize_and_center_crop(image, int(self.image_size)), dtype=np.float32)
        else:
            frame = frame.astype(np.float32)
        return torch.from_numpy(frame / 255.0).permute(2, 0, 1).contiguous()

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

    @staticmethod
    def _normalize_string_list(value: Sequence[str] | str | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @staticmethod
    def _normalize_sequence_list(value: Sequence[str | dict] | str) -> list[str | dict]:
        if isinstance(value, str):
            return [value]
        return list(value)
