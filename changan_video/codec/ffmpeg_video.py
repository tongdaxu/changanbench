from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from changan_video.codec.abs import VideoCodecIface


class FFmpegVideoCodec(VideoCodecIface):
    """FFmpeg-backed H.264/H.265/H.266 baseline video codec.

    Input and output tensors use shape (B, C, T, H, W) and range [0, 1].
    Encoding first tries the package's PyAV writers when requested, then falls
    back to the configured FFmpeg binary for codecs such as libx265/libvvenc.
    """

    def __init__(
        self,
        codec: str,
        crf: int | None = None,
        qp: int | None = None,
        bit_rate: int | None = None,
        preset: str | None = "medium",
        fps: int | float = 10,
        pix_fmt: str = "yuv420p",
        container_ext: str | None = None,
        ffmpeg_path: str | None = None,
        encode_backend: str = "auto",
        gop_size: int | None = None,
        max_b_frames: int | None = None,
        keep_temp: bool = False,
        dataset_zero_mean: bool | None = None,
        metrics_zero_mean: bool | None = None,
    ):
        super().__init__()
        if encode_backend not in {"auto", "pyav", "ffmpeg"}:
            raise ValueError("encode_backend must be one of: auto, pyav, ffmpeg")
        self.codec = codec
        self.crf = crf
        self.qp = qp
        self.bit_rate = bit_rate
        self.preset = preset
        self.fps = fps
        self.pix_fmt = pix_fmt
        self.container_ext = container_ext or self._default_ext(codec)
        self.ffmpeg_path = self._resolve_ffmpeg(ffmpeg_path)
        self.encode_backend = encode_backend
        self.gop_size = gop_size
        self.max_b_frames = max_b_frames
        self.keep_temp = keep_temp
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

        rec_tensor = torch.stack(recons, dim=0).to(device=device, dtype=dtype)
        bpp_tensor = torch.tensor(bpps, dtype=torch.float32, device=device)
        return rec_tensor, bpp_tensor

    def _process_one(self, video: torch.Tensor) -> tuple[torch.Tensor, float]:
        frames = self._tensor_to_frames(video)
        _, height, width, _ = frames.shape
        with tempfile.TemporaryDirectory(prefix="changan_ffmpeg_") as tmp:
            tmp_dir = Path(tmp)
            output_path = tmp_dir / f"encoded{self.container_ext}"
            try:
                bytes_written = self._encode_with_selected_backend(
                    frames,
                    output_path,
                    width,
                    height,
                    tmp_dir,
                )
            except Exception:
                if self.keep_temp:
                    self._persist_temp(tmp_dir, prefix="changan_ffmpeg_failed_")
                raise

            bpp = float(bytes_written * 8.0 / (frames.shape[0] * height * width))
            decoded_dir = tmp_dir / "decoded"
            self._decode_with_ffmpeg(output_path, decoded_dir)
            rec = self._read_decoded_frames(decoded_dir, frames.shape[0], height, width)
            if self.keep_temp:
                self._persist_temp(tmp_dir, prefix="changan_ffmpeg_keep_")
            return rec, bpp

    def _encode_with_selected_backend(
        self,
        frames: np.ndarray,
        output_path: Path,
        width: int,
        height: int,
        tmp_dir: Path,
    ) -> int:
        if self.encode_backend in {"auto", "pyav"}:
            try:
                return self._encode_with_pyav(frames, output_path, width, height)
            except Exception:
                if self.encode_backend == "pyav":
                    raise
        return self._encode_with_ffmpeg(frames, output_path, tmp_dir)

    def _encode_with_pyav(
        self,
        frames: np.ndarray,
        output_path: Path,
        width: int,
        height: int,
    ) -> int:
        from changan_video.codec.h264_writer import H264Writer
        from changan_video.codec.h265_writer import H265Writer
        from changan_video.codec.h266_writer import H266Writer
        from changan_video.codec.video_writer import VideoWriteConfig

        writer_type = H264Writer
        if self.codec == "libx265":
            writer_type = H265Writer
        elif self.codec == "libvvenc":
            writer_type = H266Writer

        config = VideoWriteConfig(
            output_path=output_path,
            width=width,
            height=height,
            fps=self.fps,
            codec=self.codec,
            input_format="rgb24",
            pix_fmt=self.pix_fmt,
            crf=self.crf,
            qp=self.qp,
            preset=self.preset,
            bit_rate=self.bit_rate,
            gop_size=self.gop_size,
            max_b_frames=self.max_b_frames,
        )
        with writer_type(config) as writer:
            for frame in frames:
                writer.write(frame)
            stats = writer.stats
        if stats.bytes_written is None:
            raise RuntimeError(f"Encoded file was not written: {output_path}")
        return int(stats.bytes_written)

    def _encode_with_ffmpeg(self, frames: np.ndarray, output_path: Path, tmp_dir: Path) -> int:
        input_dir = tmp_dir / "input"
        self._write_frames(frames, input_dir)
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-framerate",
            str(self.fps),
            "-i",
            str(input_dir / "frame_%05d.png"),
            "-c:v",
            self.codec,
        ]
        if self.preset:
            cmd += ["-preset", str(self.preset)]
        if self.bit_rate is not None:
            cmd += ["-b:v", str(self.bit_rate)]
        elif self.qp is not None:
            cmd += ["-qp", str(self.qp)]
        elif self.crf is not None:
            cmd += ["-crf", str(self.crf)]
        if self.gop_size is not None:
            cmd += ["-g", str(self.gop_size)]
            if self.codec == "libx264":
                cmd += ["-keyint_min", str(self.gop_size), "-sc_threshold", "0"]
            if self.codec == "libx265":
                cmd += [
                    "-x265-params",
                    f"keyint={self.gop_size}:min-keyint={self.gop_size}:scenecut=0",
                ]
        if self.max_b_frames is not None:
            cmd += ["-bf", str(self.max_b_frames)]
        cmd += ["-pix_fmt", self.pix_fmt, str(output_path)]
        self._run(cmd)
        return output_path.stat().st_size

    def _decode_with_ffmpeg(self, video_path: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-vsync",
            "0",
            "-i",
            str(video_path),
            str(out_dir / "frame_%05d.png"),
        ]
        self._run(cmd)

    def _run(self, cmd: list[str]) -> None:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n" + proc.stdout)

    def _tensor_to_frames(self, video: torch.Tensor) -> np.ndarray:
        frames = video.permute(1, 2, 3, 0).numpy()
        frames = np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frames)

    def _read_decoded_frames(
        self,
        folder: Path,
        frame_count: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        paths = sorted(folder.glob("frame_*.png"))
        if len(paths) < frame_count:
            raise RuntimeError(f"Expected {frame_count} decoded frames, found {len(paths)} in {folder}")
        frames = []
        for path in paths[:frame_count]:
            img = Image.open(path).convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        return torch.stack(frames, dim=1)

    def _write_frames(self, frames: np.ndarray, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames, start=1):
            Image.fromarray(frame, mode="RGB").save(folder / f"frame_{idx:05d}.png")

    def _persist_temp(self, tmp_dir: Path, prefix: str) -> None:
        target = Path(tempfile.mkdtemp(prefix=prefix))
        for child in tmp_dir.iterdir():
            dest = target / child.name
            if child.is_dir():
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)
        print(f"Kept temporary codec files at {target}")

    @staticmethod
    def _default_ext(codec: str) -> str:
        if codec == "libvvenc":
            return ".mkv"
        return ".mp4"

    @staticmethod
    def _resolve_ffmpeg(ffmpeg_path: str | None) -> str:
        root = Path(__file__).resolve().parents[2]
        candidates: list[Path | str] = []
        if ffmpeg_path:
            raw = Path(ffmpeg_path)
            candidates.append(raw)
            if not raw.is_absolute():
                candidates.append(root / raw)
        candidates.extend([root / "tools" / "ffmpeg" / "bin" / "ffmpeg", "ffmpeg"])
        for candidate in candidates:
            if isinstance(candidate, Path):
                for path in _path_variants(candidate):
                    if path.exists():
                        return str(path)
            else:
                found = shutil.which(candidate)
                if found:
                    return found
        raise FileNotFoundError(
            "Could not find ffmpeg. Set ffmpeg_path or place it at tools/ffmpeg/bin/ffmpeg."
        )


def _path_variants(path: Path) -> list[Path]:
    variants = [path]
    if path.suffix.lower() != ".exe":
        variants.append(path.with_name(path.name + ".exe"))
    return variants
