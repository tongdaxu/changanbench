from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cab.codec.abs import VideoCodecIface


class ExternalVideoCodec(VideoCodecIface):
    """Subprocess-backed adapter for vendored video codec projects."""

    model_dir_name: str = ""
    output_json_name: str = "metrics.json"

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        python_path: str | None = None,
        device: str = "cuda",
        keep_temp: bool = False,
        dataset_zero_mean: bool | None = None,
        metrics_zero_mean: bool | None = None,
    ):
        super().__init__()
        self.checkpoint_path = expand_project_path(checkpoint_path) if checkpoint_path else None
        self.python_path = python_path or sys.executable
        self.device = device
        self.keep_temp = bool(keep_temp)
        self.dataset_zero_mean = dataset_zero_mean
        self.metrics_zero_mean = metrics_zero_mean

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        if x.ndim != 5:
            raise ValueError(f"Expected video tensor (B, C, T, H, W), got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected RGB channel dimension of 3, got {x.shape[1]}")

        device = x.device
        dtype = x.dtype
        x_cpu = x.detach().clamp(0.0, 1.0).cpu()
        recons = []
        bpps = []
        for item in x_cpu:
            rec, bpp = self._process_one(item)
            recons.append(rec)
            bpps.append(bpp)
        return (
            torch.stack(recons, dim=0).to(device=device, dtype=dtype),
            torch.tensor(bpps, dtype=torch.float32, device=device),
        )

    @property
    def model_dir(self) -> Path:
        return project_root() / "cab" / "models" / self.model_dir_name

    def _process_one(self, video: torch.Tensor) -> tuple[torch.Tensor, float]:
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"{self.__class__.__name__} source is missing: {self.model_dir}. "
                "Vendor the official repository under cab/models first."
            )
        for path in self._required_paths():
            if not Path(path).exists():
                raise FileNotFoundError(f"Required model file not found: {path}")

        with tempfile.TemporaryDirectory(prefix=f"cab_{self.model_dir_name}_") as tmp:
            tmp_dir = Path(tmp)
            try:
                input_dir = tmp_dir / "input"
                output_dir = tmp_dir / "output"
                write_video_frames(video, input_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_json = tmp_dir / self.output_json_name
                cmd = self._build_command(input_dir, output_dir, output_json, video)
                self._run(cmd)
                rec = read_reconstruction(output_dir, video.shape[1])
                bpp = read_bpp(output_json)
                if self.keep_temp:
                    self._persist_temp(tmp_dir)
                return rec, bpp
            except Exception:
                if self.keep_temp:
                    self._persist_temp(tmp_dir)
                raise

    def _build_command(
        self,
        input_dir: Path,
        output_dir: Path,
        output_json: Path,
        video: torch.Tensor,
    ) -> list[str]:
        raise NotImplementedError

    def _required_paths(self) -> list[str]:
        return [self.checkpoint_path] if self.checkpoint_path else []

    def _run(self, cmd: list[str]) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(project_root())
            + os.pathsep
            + str(self.model_dir)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )
        proc = subprocess.run(
            cmd,
            cwd=self.model_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError("Neural video codec command failed:\n" + " ".join(cmd) + "\n" + proc.stdout)

    def _persist_temp(self, tmp_dir: Path) -> None:
        keep_dir = Path(tempfile.mkdtemp(prefix=f"cab_{self.model_dir_name}_kept_"))
        shutil.copytree(tmp_dir, keep_dir, dirs_exist_ok=True)
        print(f"Kept neural video codec temp directory: {keep_dir}")

    def fake_input(self, *args, **kwargs):
        image_size = int(kwargs.get("image_size", 256))
        batch_size = int(kwargs.get("batch_size", 1))
        frames = int(kwargs.get("frames", 32))
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        return torch.rand(batch_size, 3, frames, image_size, image_size, device=device)

    def flops(self, x, *args, **kwargs):
        return None

    def encode_time(self, x, *args, **kwargs):
        return None

    def decode_time(self, x, *args, **kwargs):
        return None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def expand_project_path(path: str) -> str:
    expanded = Path(os.path.expanduser(os.path.expandvars(path)))
    if expanded.is_absolute():
        return str(expanded)
    return str(project_root() / expanded)


def natural_path_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def write_video_frames(video: torch.Tensor, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    array = (video.permute(1, 2, 3, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    for idx, frame in enumerate(array):
        Image.fromarray(frame, mode="RGB").save(folder / f"{idx:05d}.png")


def read_reconstruction(output_dir: Path, frame_count: int) -> torch.Tensor:
    frame_paths = sorted(output_dir.rglob("*.png"), key=natural_path_key)
    if len(frame_paths) < frame_count:
        raise FileNotFoundError(
            f"Expected at least {frame_count} reconstructed frames, found {len(frame_paths)} under {output_dir}"
        )
    frames = []
    for path in frame_paths[:frame_count]:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    return torch.stack(frames, dim=1)


def read_bpp(output_json: Path) -> float:
    data = json.loads(output_json.read_text(encoding="utf-8"))
    for key in ("bpp", "ave_all_frame_bpp", "bits_per_pixel"):
        if key in data:
            return float(data[key])
    raise ValueError(f"Could not find bpp in {output_json}")
