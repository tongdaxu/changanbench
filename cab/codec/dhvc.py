from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cab.codec.external_video import ExternalVideoCodec


class DHVCVideoCodec(ExternalVideoCodec):
    model_dir_name = "dhvc"

    def __init__(self, *, gop: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.gop = int(gop)

    def _build_command(self, input_dir: Path, output_dir: Path, output_json: Path, video: torch.Tensor) -> list[str]:
        return [
            self.python_path,
            str(Path(__file__).resolve()),
            "--run-adapter",
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--checkpoint", str(self.checkpoint_path),
            "--output-json", str(output_json),
            "--device", self.device,
            "--gop", str(self.gop),
        ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-adapter", action="store_true")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gop", type=int, default=32)
    return parser.parse_args()


def pad(x, p=64):
    h, w = x.size(2), x.size(3)
    padded_h = (h + p - 1) // p * p
    padded_w = (w + p - 1) // p * p
    left = (padded_w - w) // 2
    right = padded_w - w - left
    top = (padded_h - h) // 2
    bottom = padded_h - h - top
    return F.pad(x, (left, right, top, bottom), mode="constant", value=0)


def crop(x, size):
    padded_h, padded_w = x.size(2), x.size(3)
    h, w = size
    left = (padded_w - w) // 2
    right = padded_w - w - left
    top = (padded_h - h) // 2
    bottom = padded_h - h - top
    return F.pad(x, (-left, -right, -top, -bottom), mode="constant", value=0)


def read_frame(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def write_frame(tensor: torch.Tensor, path: Path) -> None:
    arr = (
        tensor.squeeze(0)
        .detach()
        .clamp(0.0, 1.0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)).save(path)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1] / "models" / "dhvc" / "dhvc-1.0"
    sys.path.insert(0, str(repo_root))
    from models.dhvc import dhvc_base as DHVC

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(input_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {input_dir}")

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    net = DHVC()
    snapshot = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(snapshot["state_dict"])
    net.to(device).eval()
    net.compress_mode()

    first = Image.open(frame_paths[0]).convert("RGB")
    width, height = first.size
    z_lists = None
    total_bits = 0

    with torch.no_grad():
        for frame_idx, frame_path in enumerate(frame_paths):
            x = read_frame(frame_path, device)
            x_pad = pad(x)
            if frame_idx % args.gop == 0 or z_lists is None:
                z_list = net.get_temp_bias(x_pad)
                z_lists = [z_list, z_list]

            head_info, compressed_strings = net.compress(x_pad, z_lists, get_latent=True)
            total_bits += (len(head_info) + len(compressed_strings)) * 8
            rec_pad, z_new_list = net.decompress(head_info, compressed_strings, z_lists, get_latent=True)
            z_lists = [z_lists[-1], z_new_list]
            rec = crop(rec_pad.clamp(0, 1), (height, width))
            write_frame(rec, output_dir / f"{frame_idx:05d}.png")

    bpp = total_bits / (len(frame_paths) * height * width)
    Path(args.output_json).write_text(json.dumps({"bpp": bpp}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
