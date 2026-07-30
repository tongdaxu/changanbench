#!/usr/bin/env python3
"""Serial image benchmark dispatcher for the repository's Conda environments."""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = REPO_ROOT / "config/benchmarks/image_codec_registry.yaml"
DEFAULT_METRIC_CONFIG = REPO_ROOT / "config/image_metrics.yaml"

CSV_COLUMNS = [
    "dataset",
    "codec",
    "status",
    "bpp",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "dists",
    "fid",
    "parameters_m",
    "gflops",
    "time_ms",
    "environment",
    "return_code",
    "error_summary",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    codec_config: Path
    environment: str
    enabled: bool
    note: Optional[str] = None


@dataclass(frozen=True)
class Registry:
    models: List[ModelSpec]
    default_suite: List[str]

    @property
    def by_name(self) -> Dict[str, ModelSpec]:
        return {model.name: model for model in self.models}


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def load_registry(path: Path, repo_root: Path = REPO_ROOT) -> Registry:
    path = Path(path).resolve()
    data = _read_yaml(path)
    raw_models = data.get("models")
    default_suite = data.get("default_suite")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("Registry 'models' must be a non-empty list.")
    if not isinstance(default_suite, list) or not default_suite:
        raise ValueError("Registry 'default_suite' must be a non-empty list.")

    models = []
    seen = set()
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise ValueError(f"Registry model at index {index} must be a mapping.")
        missing = [
            key
            for key in ("name", "family", "codec_config", "environment", "enabled")
            if key not in raw
        ]
        if missing:
            raise ValueError(
                f"Registry model at index {index} is missing: {', '.join(missing)}"
            )

        name = str(raw["name"])
        if name in seen:
            raise ValueError(f"Duplicate registry model name: {name}")
        seen.add(name)

        codec_config = Path(str(raw["codec_config"]))
        if not codec_config.is_absolute():
            codec_config = repo_root / codec_config
        codec_config = codec_config.resolve()
        if not codec_config.is_file():
            raise ValueError(
                f"Codec config for model '{name}' does not exist: {codec_config}"
            )
        if not isinstance(raw["enabled"], bool):
            raise ValueError(f"Registry model '{name}' has a non-boolean enabled value.")

        models.append(
            ModelSpec(
                name=name,
                family=str(raw["family"]),
                codec_config=codec_config,
                environment=str(raw["environment"]),
                enabled=raw["enabled"],
                note=str(raw["note"]) if raw.get("note") is not None else None,
            )
        )

    if len(default_suite) != len(set(default_suite)):
        raise ValueError("Registry default_suite contains duplicate model names.")
    by_name = {model.name: model for model in models}
    unknown_defaults = [name for name in default_suite if name not in by_name]
    if unknown_defaults:
        raise ValueError(
            "Registry default_suite contains unknown models: "
            + ", ".join(unknown_defaults)
        )
    disabled_defaults = [name for name in default_suite if not by_name[name].enabled]
    if disabled_defaults:
        raise ValueError(
            "Registry default_suite contains disabled models: "
            + ", ".join(disabled_defaults)
        )
    return Registry(models=models, default_suite=[str(name) for name in default_suite])


def select_models(
    registry: Registry, requested: Optional[Sequence[str]]
) -> List[ModelSpec]:
    names = list(registry.default_suite if requested is None else requested)
    if not names:
        raise ValueError("At least one model must be selected.")
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError("Duplicate model names: " + ", ".join(duplicates))
    if "all" in names:
        if names != ["all"]:
            raise ValueError("'all' cannot be combined with explicit model names.")
        return [model for model in registry.models if model.enabled]

    by_name = registry.by_name
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError("Unknown model names: " + ", ".join(unknown))
    disabled = [name for name in names if not by_name[name].enabled]
    if disabled:
        raise ValueError("Disabled model names: " + ", ".join(disabled))
    return [by_name[name] for name in names]


def dataset_name_from_config(data: Dict[str, Any]) -> str:
    explicit = data.get("datasets")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("Dataset config 'datasets' must be a list.")
        names = [str(name) for name in explicit]
    else:
        names = [
            str(name)
            for name, value in data.items()
            if isinstance(value, dict) and "type" in value
        ]
    if len(names) != 1:
        raise ValueError(
            "run_benchmark.py requires exactly one dataset definition per config; "
            f"found {len(names)}."
        )
    if names[0] not in data or not isinstance(data[names[0]], dict):
        raise ValueError(f"Dataset definition is missing: {names[0]}")
    return names[0]


def prepare_dataset_config(
    source: Path, dataset_root: Optional[str], output_dir: Path
) -> tuple:
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f"Dataset config does not exist: {source}")
    data = _read_yaml(source)
    dataset_name = dataset_name_from_config(data)
    if dataset_root is None:
        return source, dataset_name

    params = data[dataset_name].setdefault("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"Dataset params must be a mapping: {dataset_name}")
    params["root"] = dataset_root

    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    override_path = config_dir / "dataset_override.yaml"
    with override_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return override_path.resolve(), dataset_name


def _display_path(path: Path, repo_root: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_command(
    model: ModelSpec,
    dataset_config: Path,
    metric_config: Path,
    result_json: Path,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    repo_root: Path = REPO_ROOT,
    conda_executable: str = "conda",
) -> List[str]:
    return [
        conda_executable,
        "run",
        "-n",
        model.environment,
        "torchrun",
        "--nproc_per_node=1",
        "--nnodes=1",
        "--node_rank=0",
        "ddp_test.py",
        "--mode",
        "image",
        "--cache_dir",
        str(cache_dir.resolve()),
        "--image_codec_config",
        _display_path(model.codec_config, repo_root),
        "--image_dataset_config",
        _display_path(dataset_config, repo_root),
        "--image_metric_config",
        _display_path(metric_config, repo_root),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--result-json",
        str(result_json.resolve()),
    ]


def _null_metrics() -> Dict[str, None]:
    return {
        "psnr": None,
        "ssim": None,
        "ms_ssim": None,
        "lpips": None,
        "dists": None,
        "fid": None,
    }


def make_scheduler_result(
    model: ModelSpec,
    dataset_name: str,
    status: str,
    command: Sequence[str],
    return_code: Optional[int] = None,
    error_summary: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "scheduler_model": model.name,
        "codec": model.name,
        "dataset": dataset_name,
        "bpp": None,
        "metrics": _null_metrics(),
        "complexity": {
            "encode_params_m": None,
            "decode_params_m": None,
            "encode_time_ms": None,
            "decode_time_ms": None,
            "encode_gflops": None,
            "decode_gflops": None,
        },
        "environment": model.environment,
        "return_code": return_code,
        "error_summary": error_summary,
        "command": list(command),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _result_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "codec" in payload:
        return payload
    results = payload.get("results")
    if isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict):
        return results[0]
    return payload


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _sum_present(*values: Any) -> Optional[float]:
    numbers = [_number(value) for value in values]
    present = [value for value in numbers if value is not None]
    return sum(present) if present else None


def result_to_row(
    payload: Dict[str, Any],
    model: ModelSpec,
    dataset_name: str,
    status_override: Optional[str] = None,
) -> Dict[str, Any]:
    record = _result_record(payload)
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    complexity = record.get("complexity")
    complexity = complexity if isinstance(complexity, dict) else {}
    return {
        "dataset": record.get("dataset") or dataset_name,
        "codec": record.get("codec") or model.name,
        "status": status_override or payload.get("status") or record.get("status") or "success",
        "bpp": _number(record.get("bpp")),
        "psnr": _number(metrics.get("psnr")),
        "ssim": _number(metrics.get("ssim")),
        "ms_ssim": _number(metrics.get("ms_ssim", metrics.get("msssim"))),
        "lpips": _number(metrics.get("lpips")),
        "dists": _number(metrics.get("dists")),
        "fid": _number(metrics.get("fid")),
        "parameters_m": _sum_present(
            complexity.get("encode_params_m"), complexity.get("decode_params_m")
        ),
        "gflops": _sum_present(
            complexity.get("encode_gflops"), complexity.get("decode_gflops")
        ),
        "time_ms": _sum_present(
            complexity.get("encode_time_ms"), complexity.get("decode_time_ms")
        ),
        "environment": payload.get("environment") or model.environment,
        "return_code": payload.get("return_code"),
        "error_summary": payload.get("error_summary"),
    }


def _error_summary(log_path: Path, limit: int = 500) -> str:
    try:
        lines = [
            line.strip()
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        return str(exc)
    return " | ".join(lines[-5:])[-limit:]


def _command_text(command: Sequence[str], gpu: Optional[str]) -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} " if gpu is not None else ""
    return prefix + shlex.join(list(command))


def write_summaries(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    csv_path = output_dir / "benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})

    markdown_path = output_dir / "benchmark.md"
    with markdown_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Image benchmark results\n\n")
        handle.write("| " + " | ".join(CSV_COLUMNS) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(CSV_COLUMNS)) + " |\n")
        for row in rows:
            values = []
            for column in CSV_COLUMNS:
                value = row.get(column)
                text = "" if value is None else str(value)
                values.append(text.replace("|", "\\|").replace("\n", " "))
            handle.write("| " + " | ".join(values) + " |\n")


def run_benchmarks(
    args: argparse.Namespace,
    subprocess_runner=subprocess.run,
    repo_root: Path = REPO_ROOT,
) -> List[Dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    json_dir = output_dir / "json"
    log_dir = output_dir / "logs"
    cache_root = output_dir / "cache"
    for directory in (output_dir, json_dir, log_dir, cache_root):
        directory.mkdir(parents=True, exist_ok=True)

    registry = load_registry(Path(args.registry), repo_root=repo_root)
    models = select_models(registry, args.models)
    dataset_config, dataset_name = prepare_dataset_config(
        Path(args.dataset_config),
        args.dataset_root,
        output_dir,
    )
    metric_config = Path(args.metric_config).resolve()
    if not metric_config.is_file():
        raise ValueError(f"Metric config does not exist: {metric_config}")

    print(f"Output directory: {output_dir}", flush=True)
    print(f"Dataset config: {dataset_config}", flush=True)
    print(f"Models: {len(models)}", flush=True)
    print(f"GPU: {args.gpu}", flush=True)

    rows = []
    stopped = False
    total_models = len(models)
    for model_index, model in enumerate(models, start=1):
        progress = f"[{model_index}/{total_models}]"
        result_json = json_dir / f"{model.name}.json"
        log_path = log_dir / f"{model.name}.log"
        cache_dir = cache_root / model.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(
            model=model,
            dataset_config=dataset_config,
            metric_config=metric_config,
            result_json=result_json,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            repo_root=repo_root,
            conda_executable=args.conda_executable,
        )
        command_text = _command_text(command, args.gpu)
        started_at = time.perf_counter()
        print(
            f"{progress} START {model.name} | env={model.environment}",
            flush=True,
        )
        print(f"log: {log_path}", flush=True)

        if stopped:
            payload = make_scheduler_result(
                model,
                dataset_name,
                "skipped_fail_fast",
                command,
                error_summary="Skipped because an earlier model failed with --fail-fast.",
            )
            write_json(result_json, payload)
            log_path.write_text(
                command_text + "\nSkipped because --fail-fast was triggered.\n",
                encoding="utf-8",
            )
            rows.append(result_to_row(payload, model, dataset_name))
            print(f"{progress} SKIP {model.name} | fail-fast triggered", flush=True)
            continue

        existing = load_json(result_json) if args.resume else None
        existing_record = _result_record(existing) if existing else {}
        if (
            existing
            and existing.get("status") == "success"
            and (existing.get("scheduler_model") in (None, model.name))
            and existing_record.get("dataset") == dataset_name
        ):
            existing["scheduler_model"] = model.name
            existing["environment"] = model.environment
            existing.setdefault("return_code", 0)
            existing.setdefault("error_summary", None)
            existing.setdefault("command", command)
            write_json(result_json, existing)
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write("[resume] Reused successful result JSON.\n")
            rows.append(result_to_row(existing, model, dataset_name))
            print(
                f"{progress} SKIP {model.name} | existing successful result",
                flush=True,
            )
            continue

        if args.dry_run:
            payload = make_scheduler_result(
                model, dataset_name, "dry_run", command
            )
            write_json(result_json, payload)
            log_path.write_text(command_text + "\n", encoding="utf-8")
            rows.append(result_to_row(payload, model, dataset_name))
            print(f"{progress} DRY-RUN {model.name}", flush=True)
            print(f"$ {command_text}", flush=True)
            continue

        result_json.unlink(missing_ok=True)
        result_json.with_suffix(result_json.suffix + ".tmp").unlink(missing_ok=True)
        child_env = os.environ.copy()
        if args.gpu is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = args.gpu

        return_code = None
        with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
            log_handle.write("$ " + command_text + "\n")
            log_handle.flush()
            try:
                completed = subprocess_runner(
                    command,
                    cwd=str(repo_root),
                    env=child_env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                return_code = int(completed.returncode)
            except OSError as exc:
                return_code = 127
                log_handle.write(f"Failed to start benchmark command: {exc}\n")

        payload = load_json(result_json) if return_code == 0 else None
        if return_code == 0 and payload is not None:
            payload["status"] = "success"
            payload["scheduler_model"] = model.name
            payload["environment"] = model.environment
            payload["return_code"] = 0
            payload["error_summary"] = None
            payload["command"] = command
            write_json(result_json, payload)
            elapsed = time.perf_counter() - started_at
            print(
                f"{progress} SUCCESS {model.name} | elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            summary = _error_summary(log_path)
            if return_code == 0:
                summary = (
                    "ddp_test.py exited successfully but did not produce valid "
                    "--result-json output. "
                    + summary
                ).strip()
            payload = make_scheduler_result(
                model,
                dataset_name,
                "failed",
                command,
                return_code=return_code,
                error_summary=summary,
            )
            write_json(result_json, payload)
            if args.fail_fast:
                stopped = True
            elapsed = time.perf_counter() - started_at
            print(
                f"{progress} FAILED {model.name} | "
                f"exit_code={return_code} | elapsed={elapsed:.1f}s",
                flush=True,
            )
            print(f"log: {log_path}", flush=True)

        rows.append(result_to_row(payload, model, dataset_name))

    write_summaries(output_dir, rows)
    print(f"Completed: {len(rows)}", flush=True)
    print(
        f"Success: {sum(row['status'] == 'success' for row in rows)}",
        flush=True,
    )
    print(
        f"Failed: {sum(row['status'] == 'failed' for row in rows)}",
        flush=True,
    )
    print(f"Results: {output_dir / 'benchmark.csv'}", flush=True)
    print(f"Markdown: {output_dir / 'benchmark.md'}", flush=True)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run image codecs serially in their registered Conda environments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Space-separated registered model names, or 'all'. "
        "Defaults to the registry default suite.",
    )
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-config", default=str(DEFAULT_METRIC_CONFIG))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-executable", default="conda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    try:
        rows = run_benchmarks(args)
    except ValueError as exc:
        parser.error(str(exc))
    failed = any(row["status"] == "failed" for row in rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
