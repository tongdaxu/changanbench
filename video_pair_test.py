from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cab.evaluations.video_pair import (
    check_dependencies,
    evaluate_video_pair,
    fvd_options_from_config,
    metrics_from_config,
    normalize_metrics,
    progress_callback,
    str_to_bool,
    vggt_metric_from_config,
    video_path_from_config,
    write_frame_metrics_csv,
    write_summary_json,
)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="test CAB pairwise video evaluation")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    if not args.config:
        raise SystemExit("--config is required.")

    from omegaconf import OmegaConf

    config_path = Path(args.config)
    config = OmegaConf.load(config_path)
    base_dir = config_path.resolve().parent

    dataset_names = list(config.get("datasets", []))
    codec_names = list(config.get("codecs", []))
    metric_entries = list(config.get("metrics", []))
    if not dataset_names:
        raise SystemExit("Config needs at least one dataset entry.")
    if not codec_names:
        raise SystemExit("Config needs at least one codec/distorted-video entry.")
    if not metric_entries:
        raise SystemExit("Config needs at least one metric entry.")

    eval_config = _plain_container(config.get("evaluation", {})) or {}
    device = str(eval_config.get("device") or _device_from_args(args))
    limit = _optional_int(eval_config.get("limit"))
    resize_distorted = _bool_config(eval_config, "resize_distorted", False)
    allow_mismatch = _bool_config(eval_config, "allow_frame_count_mismatch", False)
    include_frames = _bool_config(eval_config, "include_frames", False)
    progress_every = int(eval_config.get("progress_every", 100))

    metric_names, zero_mean, lpips_network = metrics_from_config(
        config,
        metric_entries,
        fallback_zero_mean=False,
        fallback_lpips_network="alex",
    )
    metric_names = list(normalize_metrics(metric_names))
    check_dependencies(metric_names)

    fvd_clip_length, fvd_clip_stride, fvd_model_path = fvd_options_from_config(
        config,
        metric_entries,
        base_dir=base_dir,
    )
    vggt_metric = vggt_metric_from_config(
        config,
        metric_entries,
        base_dir=base_dir,
    )

    for dataset_name in dataset_names:
        reference = video_path_from_config(
            config[dataset_name],
            entry_name=str(dataset_name),
            role="reference",
            base_dir=base_dir,
        )
        for codec_name in codec_names:
            distorted = video_path_from_config(
                config[codec_name],
                entry_name=str(codec_name),
                role="distorted",
                base_dir=base_dir,
            )
            result = evaluate_video_pair(
                reference=reference,
                distorted=distorted,
                metrics=metric_names,
                device=device,
                limit=limit,
                zero_mean=zero_mean,
                lpips_network=lpips_network,
                fvd_clip_length=fvd_clip_length,
                fvd_clip_stride=fvd_clip_stride,
                fvd_model_path=fvd_model_path,
                vggt_metric=vggt_metric,
                resize_distorted=resize_distorted,
                allow_frame_count_mismatch=allow_mismatch,
                progress=progress_callback(progress_every),
            )
            output_dir = Path(args.cache_dir) / str(codec_name) / str(dataset_name)
            write_summary_json(result, output_dir / "summary.json", include_frames=include_frames)
            write_frame_metrics_csv(result, output_dir / "frames.csv")
            _print_result(str(codec_name), str(dataset_name), result, output_dir)


def _plain_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _device_from_args(args) -> str:
    if args.local_rank is not None:
        return f"cuda:{args.local_rank}"
    return "auto"


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _bool_config(config: dict[str, Any], key: str, default: bool) -> bool:
    if key not in config:
        return default
    return str_to_bool(config[key])


def _print_result(codec_name: str, dataset_name: str, result, output_dir: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"Results: {codec_name} on {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Frames: {result.frames}")
    if result.bits_per_pixel is not None:
        print(f"BPP: {result.bits_per_pixel:.4f}")
    for name in result.metric_names:
        summary = result.metrics[name]
        print(f"{name:12s}: {summary.mean:.4f} (+/-{summary.std:.4f})")
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'frames.csv'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
