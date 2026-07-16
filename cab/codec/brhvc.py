from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

from cab.codec.external_video import ExternalVideoCodec, expand_project_path, natural_path_key


class BRHVCVideoCodec(ExternalVideoCodec):
    model_dir_name = "kwai_nvc"

    def __init__(
        self,
        *,
        i_frame_model_path: str,
        b_frame_model_path: str,
        rate_idx: int = 0,
        q_in_ckpt: bool = False,
        gop: int = 32,
        **kwargs,
    ):
        super().__init__(checkpoint_path=None, **kwargs)
        self.i_frame_model_path = expand_project_path(i_frame_model_path)
        self.b_frame_model_path = expand_project_path(b_frame_model_path)
        self.rate_idx = int(rate_idx)
        self.q_in_ckpt = bool(q_in_ckpt)
        self.gop = int(gop)

    def _required_paths(self) -> list[str]:
        return [self.i_frame_model_path, self.b_frame_model_path]

    def _build_command(self, input_dir: Path, output_dir: Path, output_json: Path, video: torch.Tensor) -> list[str]:
        return [
            self.python_path,
            str(Path(__file__).resolve()),
            "--run-adapter",
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--i-frame-model", self.i_frame_model_path,
            "--b-frame-model", self.b_frame_model_path,
            "--output-json", str(output_json),
            "--device", self.device,
            "--rate-idx", str(self.rate_idx),
            "--q-in-ckpt", "true" if self.q_in_ckpt else "false",
            "--gop", str(self.gop),
        ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-adapter", action="store_true")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--i-frame-model", required=True)
    parser.add_argument("--b-frame-model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rate-idx", type=int, default=0)
    parser.add_argument("--q-in-ckpt", choices=("true", "false"), default="false")
    parser.add_argument("--gop", type=int, default=32)
    return parser.parse_args()


def prepare_dataset(input_dir: Path, dataset_root: Path) -> tuple[int, int, int]:
    sample_dir = dataset_root / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(input_dir.glob("*.png"), key=natural_path_key)
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {input_dir}")
    width, height = Image.open(frame_paths[0]).convert("RGB").size
    for idx, src in enumerate(frame_paths, start=1):
        shutil.copy2(src, sample_dir / f"im{idx:05d}.png")
    return len(frame_paths), width, height


def write_config(path: Path, dataset_root: Path, frame_count: int, width: int, height: int, gop: int) -> None:
    if gop != 32:
        raise ValueError(f"BRHVC only supports GOP 32 in the official script, got {gop}")
    if frame_count < gop + 1:
        raise ValueError(f"BRHVC needs at least {gop + 1} frames for GOP {gop}, got {frame_count}")
    if (frame_count - 1) % gop != 0:
        raise ValueError(f"BRHVC expects frame_count = 1 + N * GOP, got {frame_count} for GOP {gop}")
    config = {
        "root_path": str(dataset_root),
        "test_classes": {
            "CAB": {
                "test": 1,
                "base_path": ".",
                "src_type": "png",
                "sequences": {
                    "sample": {"width": width, "height": height, "frames": frame_count, "gop": gop}
                },
            }
        },
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_codec(args, config_path: Path, decoded_root: Path, result_json: Path, stream_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1] / "models" / "kwai_nvc" / "BRHVC"
    cuda = "true" if str(args.device).startswith("cuda") else "false"
    cuda_device = str(args.device).split(":", 1)[1] if ":" in str(args.device) else None
    cmd = [
        sys.executable,
        "test_video.py",
        "--i_frame_model_path", args.i_frame_model,
        "--b_frame_model_path", args.b_frame_model,
        "--rate_num", "1",
        "--q_in_ckpt", args.q_in_ckpt,
        "--i_frame_q_indexes", str(args.rate_idx),
        "--b_frame_q_indexes", str(args.rate_idx),
        "--test_config", str(config_path),
        "--cuda", cuda,
        "--worker", "1",
        "--calc_ssim", "false",
        "--write_stream", "false",
        "--stream_path", str(stream_root),
        "--save_decoded_frame", "true",
        "--decoded_frame_path", str(decoded_root),
        "--output_path", str(result_json),
        "--force_intra_period", str(args.gop),
        "--force_frame_num", str(read_frame_count(config_path)),
        "--verbose", "0",
    ]
    if cuda_device:
        cmd += ["--cuda_device", cuda_device]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def read_frame_count(config_path: Path) -> int:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return int(data["test_classes"]["CAB"]["sequences"]["sample"]["frames"])


def extract_bpp(result_json: Path) -> float:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    for ds_value in data.values():
        for seq_value in ds_value.values():
            for rate_value in seq_value.values():
                if "ave_all_frame_bpp" in rate_value:
                    return float(rate_value["ave_all_frame_bpp"])
    raise ValueError(f"Could not find ave_all_frame_bpp in {result_json}")


def copy_recon(decoded_root: Path, output_dir: Path, frame_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted((decoded_root / "sample").rglob("*.png"), key=natural_path_key)
    if len(frames) < frame_count:
        raise FileNotFoundError(f"Expected {frame_count} recon frames, found {len(frames)} under {decoded_root}")
    for idx, src in enumerate(frames[:frame_count]):
        shutil.copy2(src, output_dir / f"{idx:05d}.png")


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="cab_brhvc_") as tmp:
        tmp_dir = Path(tmp)
        dataset_root = tmp_dir / "dataset"
        frame_count, width, height = prepare_dataset(Path(args.input_dir), dataset_root)
        config_path = tmp_dir / "dataset_config.json"
        write_config(config_path, dataset_root, frame_count, width, height, args.gop)
        decoded_root = tmp_dir / "decoded"
        result_json = tmp_dir / "result.json"
        run_codec(args, config_path, decoded_root, result_json, tmp_dir / "streams")
        copy_recon(decoded_root, Path(args.output_dir), frame_count)
        bpp = extract_bpp(result_json)
        Path(args.output_json).write_text(json.dumps({"bpp": bpp}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
