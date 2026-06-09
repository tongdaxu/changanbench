from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cab.codec.abs import VideoCodecIface


class _DCVCSubprocessCodec(VideoCodecIface):
    """CAB adapter for vendored DCVC video scripts.

    The DCVC-family repositories all expose script-oriented evaluators and many
    of them import a top-level module named ``src``. Running each codec in a
    subprocess with its own working directory keeps those import namespaces
    isolated while preserving CAB's ``forward(x) -> (xhat, bpp)`` interface.
    """

    family_dir_name: str = ""
    output_json_name: str = "output.json"

    def __init__(
        self,
        *,
        python_path: str | None = None,
        device: str = "cuda",
        worker: int = 1,
        force_intra: bool = False,
        intra_period: int | None = None,
        write_stream: bool = False,
        keep_temp: bool = False,
        verbose: int = 0,
        dataset_zero_mean: bool | None = None,
        metrics_zero_mean: bool | None = None,
    ):
        super().__init__()
        self.python_path = python_path or sys.executable
        self.device = device
        self.worker = int(worker)
        self.force_intra = bool(force_intra)
        self.intra_period = intra_period
        self.write_stream = bool(write_stream)
        self.keep_temp = bool(keep_temp)
        self.verbose = int(verbose)
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
    def family_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "models" / "dcvc_family" / self.family_dir_name

    def _process_one(self, video: torch.Tensor) -> tuple[torch.Tensor, float]:
        with tempfile.TemporaryDirectory(prefix=f"cab_{self.family_dir_name}_") as tmp:
            tmp_dir = Path(tmp)
            try:
                self._write_video_frames(video, tmp_dir / "dataset" / "sample")
                config_path = self._write_config(tmp_dir, video)
                output_path = tmp_dir / self.output_json_name
                cmd = self._build_command(tmp_dir, config_path, output_path, video)
                self._run(cmd)
                bpp = self._read_bpp(output_path)
                rec = self._read_reconstruction(tmp_dir, video.shape[1])
                if self.keep_temp:
                    self._persist_temp(tmp_dir)
                return rec, bpp
            except Exception:
                if self.keep_temp:
                    self._persist_temp(tmp_dir)
                raise

    def _write_config(self, tmp_dir: Path, video: torch.Tensor) -> Path:
        _, frames, height, width = video.shape
        config = self._config_dict(tmp_dir, frames, height, width)
        path = tmp_dir / "dataset_config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return path

    def _config_dict(self, tmp_dir: Path, frames: int, height: int, width: int) -> dict[str, Any]:
        root = str(tmp_dir / "dataset")
        return {
            "root_path": root,
            "test_classes": {
                "CAB": {
                    "test": 1,
                    "base_path": ".",
                    "src_type": "png",
                    "sequences": {
                        "sample": self._sequence_info(frames, height, width),
                    },
                }
            },
        }

    def _sequence_info(self, frames: int, height: int, width: int) -> dict[str, int]:
        info = {"width": width, "height": height, "frames": frames}
        if self.intra_period is not None:
            info["intra_period"] = int(self.intra_period)
            info["gop"] = int(self.intra_period)
        else:
            info["intra_period"] = frames
            info["gop"] = frames
        return info

    def _build_command(self, tmp_dir: Path, config_path: Path, output_path: Path, video: torch.Tensor) -> list[str]:
        raise NotImplementedError

    def _run(self, cmd: list[str]) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.family_dir) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            cmd,
            cwd=self.family_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError("DCVC command failed:\n" + " ".join(cmd) + "\n" + proc.stdout)

    def _cuda_args(self) -> list[str]:
        if not str(self.device).startswith("cuda"):
            return ["--cuda", "false"]
        args = ["--cuda", "true"]
        if ":" in str(self.device):
            args += ["--cuda_device", str(self.device).split(":", 1)[1]]
        return args

    def _common_script_args(self, tmp_dir: Path, config_path: Path, output_path: Path) -> list[str]:
        args = [
            "--test_config", str(config_path),
            "--worker", str(self.worker),
            "--write_stream", _bool_arg(self.write_stream),
            "--save_decoded_frame", "true",
            "--output_path", str(output_path),
            "--verbose", str(self.verbose),
        ]
        if self.force_intra:
            args += ["--force_intra", "true"]
        if self.intra_period is not None:
            args += ["--force_intra_period", str(int(self.intra_period))]
        args += self._cuda_args()
        return args

    def _read_bpp(self, output_path: Path) -> float:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        candidates = []

        def visit(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in {"ave_all_frame_bpp", "ave_all_frame_quality"}:
                        candidates.append((key, value))
                    visit(value)
            elif isinstance(obj, list):
                for item in obj:
                    visit(item)

        visit(data)
        for key, value in candidates:
            if key == "ave_all_frame_bpp":
                return float(value)
        raise ValueError(f"Could not find ave_all_frame_bpp in {output_path}")

    def _read_reconstruction(self, tmp_dir: Path, frame_count: int) -> torch.Tensor:
        frame_paths = sorted(
            (path for path in tmp_dir.rglob("*.png") if self._is_reconstruction_path(path)),
            key=_natural_path_key,
        )
        if len(frame_paths) < frame_count:
            raise FileNotFoundError(
                f"Expected at least {frame_count} reconstructed frames, found {len(frame_paths)} under {tmp_dir}"
            )
        frame_paths = frame_paths[:frame_count]
        frames = []
        for path in frame_paths:
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        return torch.stack(frames, dim=1)

    def _is_reconstruction_path(self, path: Path) -> bool:
        parts = set(path.parts)
        if "dataset" in parts:
            return False
        name = path.name
        return (
            "decoded" in parts
            or "recon" in parts
            or "recon_bin" in parts
            or name.startswith("recon_frame_")
            or name.startswith("im")
        )

    def _persist_temp(self, tmp_dir: Path) -> None:
        keep_dir = Path(tempfile.mkdtemp(prefix=f"cab_{self.family_dir_name}_kept_"))
        shutil.copytree(tmp_dir, keep_dir, dirs_exist_ok=True)
        print(f"Kept DCVC temp directory: {keep_dir}")

    @staticmethod
    def _write_video_frames(video: torch.Tensor, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        array = (video.permute(1, 2, 3, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        for idx, frame in enumerate(array, start=1):
            Image.fromarray(frame, mode="RGB").save(folder / f"im{idx:05d}.png")


class DCVCVideoCodec(_DCVCSubprocessCodec):
    family_dir_name = "dcvc"
    output_json_name = "dcvc_output.json"

    def __init__(
        self,
        *,
        i_frame_model_paths: list[str],
        model_paths: list[str],
        model_index: int = 0,
        i_frame_model_name: str = "cheng2020-anchor",
        model_type: str = "psnr",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.i_frame_model_paths = [_expand_path(path) for path in i_frame_model_paths]
        self.model_paths = [_expand_path(path) for path in model_paths]
        self.model_index = int(model_index)
        self.i_frame_model_name = i_frame_model_name
        self.model_type = model_type

    def _config_dict(self, tmp_dir: Path, frames: int, height: int, width: int) -> dict[str, Any]:
        return {
            "CAB": {
                "base_path": str(tmp_dir / "dataset"),
                "sequences": {
                    "sample": {
                        "frames": frames,
                        "gop": int(self.intra_period) if self.intra_period is not None else frames,
                    }
                },
            }
        }

    def _build_command(self, tmp_dir: Path, config_path: Path, output_path: Path, video: torch.Tensor) -> list[str]:
        i_path = self.i_frame_model_paths[self.model_index]
        p_path = self.model_paths[self.model_index]
        return [
            self.python_path,
            "test_video.py",
            "--i_frame_model_name", self.i_frame_model_name,
            "--i_frame_model_path", i_path,
            "--model_path", p_path,
            "--test_config", str(config_path),
            "--worker", str(self.worker),
            "--write_stream", _bool_arg(self.write_stream),
            "--write_recon_frame", "true",
            "--recon_bin_path", str(tmp_dir / "recon_bin"),
            "--output_json_result_path", str(output_path),
            "--model_type", self.model_type,
        ] + self._cuda_args()


class DCVCTCMVideoCodec(_DCVCSubprocessCodec):
    family_dir_name = "dcvc_tcm"

    def __init__(
        self,
        *,
        i_frame_model_paths: list[str],
        model_paths: list[str],
        model_index: int = 0,
        i_frame_model_name: str = "IntraNoAR",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.i_frame_model_paths = [_expand_path(path) for path in i_frame_model_paths]
        self.model_paths = [_expand_path(path) for path in model_paths]
        self.model_index = int(model_index)
        self.i_frame_model_name = i_frame_model_name

    def _config_dict(self, tmp_dir: Path, frames: int, height: int, width: int) -> dict[str, Any]:
        return {
            "CAB": {
                "test": 1,
                "base_path": str(tmp_dir / "dataset"),
                "sequences": {
                    "sample": {
                        "frames": frames,
                        "gop": int(self.intra_period) if self.intra_period is not None else frames,
                    }
                },
            }
        }

    def _build_command(self, tmp_dir: Path, config_path: Path, output_path: Path, video: torch.Tensor) -> list[str]:
        i_path = self.i_frame_model_paths[self.model_index]
        p_path = self.model_paths[self.model_index]
        return [
            self.python_path,
            "test_video.py",
            "--i_frame_model_name", self.i_frame_model_name,
            "--i_frame_model_path", i_path,
            "--model_path", p_path,
            "--stream_path", str(tmp_dir / "streams"),
            "--decoded_frame_path", str(tmp_dir / "decoded"),
        ] + self._common_script_args(tmp_dir, config_path, output_path)


class DCVCHEMVideoCodec(_DCVCSubprocessCodec):
    family_dir_name = "dcvc_hem"

    def __init__(
        self,
        *,
        i_frame_model_path: str,
        model_path: str,
        i_frame_q_scale: float | None = None,
        p_frame_y_q_scale: float | None = None,
        p_frame_mv_y_q_scale: float | None = None,
        rate_num: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.i_frame_model_path = _expand_path(i_frame_model_path)
        self.model_path = _expand_path(model_path)
        self.i_frame_q_scale = i_frame_q_scale
        self.p_frame_y_q_scale = p_frame_y_q_scale
        self.p_frame_mv_y_q_scale = p_frame_mv_y_q_scale
        self.rate_num = int(rate_num)

    def _build_command(self, tmp_dir: Path, config_path: Path, output_path: Path, video: torch.Tensor) -> list[str]:
        cmd = [
            self.python_path,
            "test_video.py",
            "--i_frame_model_path", self.i_frame_model_path,
            "--model_path", self.model_path,
            "--rate_num", str(self.rate_num),
            "--force_root_path", str(tmp_dir / "dataset"),
            "--stream_path", str(tmp_dir / "streams"),
            "--decoded_frame_path", str(tmp_dir / "decoded"),
        ] + self._common_script_args(tmp_dir, config_path, output_path)
        if self.i_frame_q_scale is not None:
            cmd += ["--i_frame_q_scales", str(self.i_frame_q_scale)]
        if self.p_frame_y_q_scale is not None and self.p_frame_mv_y_q_scale is not None:
            cmd += [
                "--p_frame_y_q_scales", str(self.p_frame_y_q_scale),
                "--p_frame_mv_y_q_scales", str(self.p_frame_mv_y_q_scale),
            ]
        return cmd


class DCVCDCVideoCodec(_DCVCSubprocessCodec):
    family_dir_name = "dcvc_dc"

    def __init__(
        self,
        *,
        i_frame_model_path: str,
        p_frame_model_path: str,
        i_frame_q_index: int | None = None,
        p_frame_q_index: int | None = None,
        rate_num: int = 4,
        calc_ssim: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.i_frame_model_path = _expand_path(i_frame_model_path)
        self.p_frame_model_path = _expand_path(p_frame_model_path)
        self.i_frame_q_index = i_frame_q_index
        self.p_frame_q_index = p_frame_q_index
        self.rate_num = int(rate_num)
        self.calc_ssim = bool(calc_ssim)

    def _build_command(self, tmp_dir: Path, config_path: Path, output_path: Path, video: torch.Tensor) -> list[str]:
        cmd = [
            self.python_path,
            "test_video.py",
            "--i_frame_model_path", self.i_frame_model_path,
            "--p_frame_model_path", self.p_frame_model_path,
            "--rate_num", str(self.rate_num),
            "--yuv420", "false",
            "--calc_ssim", _bool_arg(self.calc_ssim),
            "--stream_path", str(tmp_dir / "streams"),
            "--decoded_frame_path", str(tmp_dir / "decoded"),
        ] + self._common_script_args(tmp_dir, config_path, output_path)
        if self.i_frame_q_index is not None:
            cmd += ["--i_frame_q_indexes", str(self.i_frame_q_index)]
        if self.p_frame_q_index is not None:
            cmd += ["--p_frame_q_indexes", str(self.p_frame_q_index)]
        return cmd


class DCVCFMVideoCodec(_DCVCSubprocessCodec):
    family_dir_name = "dcvc_fm"

    def __init__(
        self,
        *,
        model_path_i: str,
        model_path_p: str,
        q_index_i: int | None = None,
        q_index_p: int | None = None,
        rate_num: int = 4,
        rate_gop_size: int = 8,
        reset_interval: int = 32,
        float16: bool = False,
        calc_ssim: bool = False,
        **kwargs,
    ):
        kwargs.pop("write_stream", None)
        super().__init__(write_stream=True, **kwargs)
        self.model_path_i = _expand_path(model_path_i)
        self.model_path_p = _expand_path(model_path_p)
        self.q_index_i = q_index_i
        self.q_index_p = q_index_p
        self.rate_num = int(rate_num)
        self.rate_gop_size = int(rate_gop_size)
        self.reset_interval = int(reset_interval)
        self.float16 = bool(float16)
        self.calc_ssim = bool(calc_ssim)

    def _build_command(self, tmp_dir: Path, config_path: Path, output_path: Path, video: torch.Tensor) -> list[str]:
        cmd = [
            self.python_path,
            "test_video.py",
            "--model_path_i", self.model_path_i,
            "--model_path_p", self.model_path_p,
            "--rate_num", str(self.rate_num),
            "--force_root_path", str(tmp_dir / "dataset"),
            "--stream_path", str(tmp_dir / "streams"),
            "--rate_gop_size", str(self.rate_gop_size),
            "--reset_interval", str(self.reset_interval),
            "--float16", _bool_arg(self.float16),
            "--calc_ssim", _bool_arg(self.calc_ssim),
        ] + self._common_script_args(tmp_dir, config_path, output_path)
        if self.q_index_i is not None:
            cmd += ["--q_indexes_i", str(self.q_index_i)]
        if self.q_index_p is not None:
            cmd += ["--q_indexes_p", str(self.q_index_p)]
        return cmd

    def _cuda_args(self) -> list[str]:
        if not str(self.device).startswith("cuda"):
            return ["--cuda", "false"]
        args = ["--cuda", "true"]
        if ":" in str(self.device):
            args += ["--cuda_idx", str(self.device).split(":", 1)[1]]
        return args


class DCVCRTVideoCodec(_DCVCSubprocessCodec):
    family_dir_name = "dcvc_rt"

    def __init__(
        self,
        *,
        model_path_i: str,
        model_path_p: str,
        qp_i: int | None = None,
        qp_p: int | None = None,
        rate_num: int = 4,
        reset_interval: int = 64,
        force_zero_thres: float | None = 0.12,
        calc_ssim: bool = False,
        **kwargs,
    ):
        kwargs.pop("write_stream", None)
        super().__init__(write_stream=True, **kwargs)
        self.model_path_i = _expand_path(model_path_i)
        self.model_path_p = _expand_path(model_path_p)
        self.qp_i = qp_i
        self.qp_p = qp_p
        self.rate_num = int(rate_num)
        self.reset_interval = int(reset_interval)
        self.force_zero_thres = force_zero_thres
        self.calc_ssim = bool(calc_ssim)

    def _build_command(self, tmp_dir: Path, config_path: Path, output_path: Path, video: torch.Tensor) -> list[str]:
        cmd = [
            self.python_path,
            "test_video.py",
            "--model_path_i", self.model_path_i,
            "--model_path_p", self.model_path_p,
            "--rate_num", str(self.rate_num),
            "--force_root_path", str(tmp_dir / "dataset"),
            "--stream_path", str(tmp_dir / "streams"),
            "--reset_interval", str(self.reset_interval),
            "--check_existing", "false",
            "--calc_ssim", _bool_arg(self.calc_ssim),
        ] + self._common_script_args(tmp_dir, config_path, output_path)
        if self.force_zero_thres is not None:
            cmd += ["--force_zero_thres", str(self.force_zero_thres)]
        if self.qp_i is not None:
            cmd += ["--qp_i", str(self.qp_i)]
        if self.qp_p is not None:
            cmd += ["--qp_p", str(self.qp_p)]
        return cmd

    def _cuda_args(self) -> list[str]:
        if not str(self.device).startswith("cuda"):
            return ["--cuda", "false"]
        args = ["--cuda", "true"]
        if ":" in str(self.device):
            args += ["--cuda_idx", str(self.device).split(":", 1)[1]]
        return args


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def _expand_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def _natural_path_key(path: Path):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    ]
