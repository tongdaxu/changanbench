"""Download Cosmos image tokenizer weights used by CAB image codec configs.

The script downloads the two JIT image tokenizer checkpoints referenced by
config/image_codecs/cosmos_discrete_q*.yaml into:

    /data9-2/BenchmarkData/weights/cosmos/<model-name>/

Use HF_TOKEN when the Hugging Face repository requires authentication.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


MODELS = {
    "Cosmos-0.1-Tokenizer-DI16x16": "nvidia/Cosmos-0.1-Tokenizer-DI16x16",
    "Cosmos-0.1-Tokenizer-DI8x8": "nvidia/Cosmos-0.1-Tokenizer-DI8x8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Cosmos ImageTokenizer weights")
    parser.add_argument(
        "--output-root",
        default="/data9-2/BenchmarkData/weights",
        help="Root directory for image codec weights.",
    )
    parser.add_argument(
        "--codec-dir",
        default="cosmos",
        help="Subdirectory under output-root for Cosmos tokenizer weights.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with "
            "`pip install huggingface_hub` in the evaluation environment."
        ) from exc

    root = Path(args.output_root) / args.codec_dir
    root.mkdir(parents=True, exist_ok=True)

    for model_name, repo_id in MODELS.items():
        local_dir = root / model_name
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["encoder.jit", "decoder.jit"],
            token=args.token,
            revision=args.revision,
        )

        missing = [
            filename
            for filename in ("encoder.jit", "decoder.jit")
            if not (local_dir / filename).exists()
        ]
        if missing:
            raise RuntimeError(
                f"{repo_id} did not provide expected files: {', '.join(missing)}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
