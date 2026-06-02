from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from changan_video.evaluations.video import (
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
    parser = argparse.ArgumentParser(description="test changan bench with DDP")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--local-rank', type=int, default=None)
    parser.add_argument('--zero-mean', type=str_to_bool, default=False, help='Only needed for visualization')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    if not args.config:
        raise SystemExit("--config is required.")
    run_config(args)


def run_config(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("omegaconf") is None:
        raise SystemExit(
            "Missing dependency: omegaconf. Activate the evaluation environment first."
        )

    from omegaconf import OmegaConf

    config_path = Path(args.config)
    config = OmegaConf.load(config_path)
    config_dir = config_path.resolve().parent

    datasets_name = list(config["datasets"])
    codecs_name = list(config["codecs"])
    metrics_name = list(config["metrics"])

    metric_names, zero_mean, lpips_network = metrics_from_config(
        config,
        metrics_name,
        fallback_zero_mean=args.zero_mean,
        fallback_lpips_network="alex",
    )
    fvd_clip_length, fvd_clip_stride, fvd_model_path = fvd_options_from_config(
        config,
        metrics_name,
        base_dir=config_dir,
    )
    vggt_metric = vggt_metric_from_config(
        config,
        metrics_name,
        base_dir=config_dir,
    )
    metric_names = normalize_metrics(metric_names)
    check_dependencies(metric_names)

    for codec_name in codecs_name:
        distorted = video_path_from_config(
            config[codec_name],
            entry_name=str(codec_name),
            role="distorted",
            base_dir=config_dir,
        )
        for dataset_name in datasets_name:
            reference = video_path_from_config(
                config[dataset_name],
                entry_name=str(dataset_name),
                role="reference",
                base_dir=config_dir,
            )
            result = _evaluate(
                reference,
                distorted,
                metric_names,
                zero_mean,
                lpips_network,
                fvd_clip_length,
                fvd_clip_stride,
                fvd_model_path,
                vggt_metric,
            )
            cache_file_name = Path(args.cache_dir) / str(codec_name) / str(dataset_name)
            _save_outputs(result, cache_file_name)
            print_results(str(codec_name), str(dataset_name), result)


def _evaluate(
    reference: str,
    distorted: str,
    metric_names,
    zero_mean: bool,
    lpips_network: str,
    fvd_clip_length: int,
    fvd_clip_stride: int,
    fvd_model_path: str | None,
    vggt_metric,
):
    return evaluate_video_pair(
        reference=reference,
        distorted=distorted,
        metrics=metric_names,
        device="auto",
        limit=None,
        zero_mean=zero_mean,
        lpips_network=lpips_network,
        fvd_clip_length=fvd_clip_length,
        fvd_clip_stride=fvd_clip_stride,
        fvd_model_path=fvd_model_path,
        vggt_metric=vggt_metric,
        resize_distorted=False,
        allow_frame_count_mismatch=False,
        progress=progress_callback(100),
    )


def _save_outputs(result, cache_file_name: Path) -> None:
    cache_file_name.mkdir(parents=True, exist_ok=True)
    write_frame_metrics_csv(result, cache_file_name / "frame_metrics.csv")
    write_summary_json(result, cache_file_name / "summary.json")


def print_results(codec_name: str, dataset_name: str, result) -> None:
    print(f"\n{'=' * 60}")
    print(f"Results: {codec_name} on {dataset_name}")
    print(f"{'=' * 60}")
    if result.bits_per_pixel is not None:
        print(f"BPP: {result.bits_per_pixel:.4f}")
    for metric_name in result.metric_names:
        summary = result.metrics[metric_name]
        print(f"{metric_name:12s}: {summary.mean:.4f} (+/-{summary.std:.4f})")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
