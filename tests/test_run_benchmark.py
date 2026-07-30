import argparse
import csv
import json
import subprocess
from pathlib import Path

import pytest
import yaml

import run_benchmark


REPO_ROOT = Path(run_benchmark.__file__).resolve().parent
REGISTRY_PATH = REPO_ROOT / "config/benchmarks/image_codec_registry.yaml"
METRIC_CONFIG = REPO_ROOT / "config/image_metrics.yaml"


def _dataset_config(tmp_path, name="sample"):
    path = tmp_path / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                name: {
                    "type": "example.Dataset",
                    "params": {"root": "original/images", "zero_mean": None},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _args(tmp_path, models, **overrides):
    values = {
        "models": models,
        "dataset_config": str(_dataset_config(tmp_path)),
        "dataset_root": None,
        "output_dir": str(tmp_path / "output"),
        "metric_config": str(METRIC_CONFIG),
        "registry": str(REGISTRY_PATH),
        "batch_size": 1,
        "num_workers": 0,
        "gpu": "0",
        "conda_executable": "conda",
        "resume": False,
        "dry_run": False,
        "fail_fast": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_success_result(command, model_name=None):
    result_path = Path(command[command.index("--result-json") + 1])
    codec_config = command[command.index("--image_codec_config") + 1]
    codec = model_name or Path(codec_config).stem
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "success",
                "codec": codec,
                "dataset": "sample",
                "bpp": 0.25,
                "metrics": {
                    "psnr": 31.0,
                    "ssim": 0.91,
                    "ms_ssim": 0.96,
                    "lpips": 0.1,
                    "dists": 0.2,
                    "fid": 12.0,
                },
                "complexity": {
                    "encode_params_m": 1.0,
                    "decode_params_m": 2.0,
                    "encode_gflops": 3.0,
                    "decode_gflops": 4.0,
                    "encode_time_ms": 5.0,
                    "decode_time_ms": 6.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_registry_and_model_selection():
    registry = run_benchmark.load_registry(REGISTRY_PATH)
    assert len(registry.models) == 20
    default_models = run_benchmark.select_models(registry, None)
    assert [model.name for model in default_models] == [
        "bsq_q0",
        "fsq_q0",
        "jpeg_q22",
        "var_q0",
        "flowmo_q0",
        "ssdd_q0",
        "ibq_q0",
        "perco_q0",
        "tatok_q0",
        "infinity_q0",
    ]
    assert registry.by_name["tatok_q0"].environment == "infinity"
    assert sum(model.environment == "illm" for model in default_models) == 8
    assert sum(model.environment == "infinity" for model in default_models) == 2
    assert [model.name for model in run_benchmark.select_models(registry, ["all"])] == [
        model.name for model in registry.models if model.enabled
    ]


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["not_a_codec"], "Unknown model names"),
        (["bsq_q0", "bsq_q0"], "Duplicate model names"),
        (["all", "bsq_q0"], "cannot be combined"),
    ],
)
def test_invalid_model_selection(names, message):
    registry = run_benchmark.load_registry(REGISTRY_PATH)
    with pytest.raises(ValueError, match=message):
        run_benchmark.select_models(registry, names)


def test_environment_command_construction(tmp_path):
    registry = run_benchmark.load_registry(REGISTRY_PATH)
    model = registry.by_name["infinity_q0"]
    command = run_benchmark.build_command(
        model=model,
        dataset_config=_dataset_config(tmp_path),
        metric_config=METRIC_CONFIG,
        result_json=tmp_path / "result.json",
        cache_dir=tmp_path / "cache",
        batch_size=1,
        num_workers=0,
    )
    assert command[:6] == [
        "conda",
        "run",
        "-n",
        "infinity",
        "torchrun",
        "--nproc_per_node=1",
    ]
    assert command[command.index("--image_codec_config") + 1] == (
        "config/image_codecs/infinity_q0.yaml"
    )
    assert command[command.index("--result-json") + 1] == str(
        (tmp_path / "result.json").resolve()
    )


def test_dataset_root_override_does_not_modify_source(tmp_path):
    source = _dataset_config(tmp_path)
    original = source.read_text(encoding="utf-8")
    override, dataset_name = run_benchmark.prepare_dataset_config(
        source,
        "/data/company/images",
        tmp_path / "output",
    )
    assert dataset_name == "sample"
    assert source.read_text(encoding="utf-8") == original
    data = yaml.safe_load(override.read_text(encoding="utf-8"))
    assert data["sample"]["params"]["root"] == "/data/company/images"
    assert override.parent == (tmp_path / "output" / "config").resolve()


def test_dry_run_creates_per_model_artifacts_and_summaries(tmp_path, capsys):
    args = _args(tmp_path, ["bsq_q0", "jpeg_q22"], dry_run=True)

    def unexpected_subprocess(*args, **kwargs):
        raise AssertionError("dry-run must not invoke subprocess")

    rows = run_benchmark.run_benchmarks(args, subprocess_runner=unexpected_subprocess)
    assert [row["status"] for row in rows] == ["dry_run", "dry_run"]
    for model in ("bsq_q0", "jpeg_q22"):
        assert (tmp_path / "output/json" / f"{model}.json").is_file()
        assert (tmp_path / "output/logs" / f"{model}.log").is_file()
        assert (tmp_path / "output/cache" / model).is_dir()
    assert (tmp_path / "output/benchmark.csv").is_file()
    assert (tmp_path / "output/benchmark.md").is_file()
    output = capsys.readouterr().out
    assert f"Output directory: {(tmp_path / 'output').resolve()}" in output
    assert "Models: 2" in output
    assert "GPU: 0" in output
    assert "[1/2] START bsq_q0 | env=illm" in output
    assert f"log: {(tmp_path / 'output/logs/bsq_q0.log').resolve()}" in output
    assert "[1/2] DRY-RUN bsq_q0" in output
    assert "$ CUDA_VISIBLE_DEVICES=0 conda run -n illm torchrun" in output
    assert "Completed: 2" in output
    assert "Success: 0" in output
    assert "Failed: 0" in output
    assert f"Results: {(tmp_path / 'output/benchmark.csv').resolve()}" in output
    assert f"Markdown: {(tmp_path / 'output/benchmark.md').resolve()}" in output


def test_dry_run_uses_infinity_for_tatok_and_infinity(tmp_path):
    args = _args(tmp_path, ["tatok_q0", "infinity_q0"], dry_run=True)

    def unexpected_subprocess(*args, **kwargs):
        raise AssertionError("dry-run must not invoke subprocess")

    run_benchmark.run_benchmarks(args, subprocess_runner=unexpected_subprocess)
    for model in ("tatok_q0", "infinity_q0"):
        log = (tmp_path / "output/logs" / f"{model}.log").read_text(
            encoding="utf-8"
        )
        assert "conda run -n infinity torchrun" in log


def test_failure_continues_to_next_model(tmp_path, capsys):
    args = _args(tmp_path, ["bsq_q0", "jpeg_q22"])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            kwargs["stdout"].write("first model failed\n")
            return subprocess.CompletedProcess(command, 9)
        _write_success_result(command)
        return subprocess.CompletedProcess(command, 0)

    rows = run_benchmark.run_benchmarks(args, subprocess_runner=fake_run)
    assert len(calls) == 2
    assert [row["status"] for row in rows] == ["failed", "success"]
    assert rows[0]["return_code"] == 9
    assert "first model failed" in rows[0]["error_summary"]
    output = capsys.readouterr().out
    assert "[1/2] FAILED bsq_q0 | exit_code=9 | elapsed=" in output
    assert f"log: {(tmp_path / 'output/logs/bsq_q0.log').resolve()}" in output
    assert "[2/2] SUCCESS jpeg_q22 | elapsed=" in output
    assert "Completed: 2" in output
    assert "Success: 1" in output
    assert "Failed: 1" in output


def test_resume_reuses_successful_json(tmp_path, capsys):
    args = _args(tmp_path, ["bsq_q0"], resume=True)
    result_path = tmp_path / "output/json/bsq_q0.json"
    result_path.parent.mkdir(parents=True)
    payload = {
        "status": "success",
        "scheduler_model": "bsq_q0",
        "codec": "bsq_q0",
        "dataset": "sample",
        "bpp": 0.5,
        "metrics": {},
        "complexity": {},
        "return_code": 0,
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    def unexpected_subprocess(*args, **kwargs):
        raise AssertionError("resume must not rerun a successful model")

    rows = run_benchmark.run_benchmarks(args, subprocess_runner=unexpected_subprocess)
    assert rows[0]["status"] == "success"
    assert rows[0]["bpp"] == 0.5
    assert "Reused successful result" in (
        tmp_path / "output/logs/bsq_q0.log"
    ).read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "[1/1] START bsq_q0 | env=illm" in output
    assert "[1/1] SKIP bsq_q0 | existing successful result" in output
    assert "Completed: 1" in output
    assert "Success: 1" in output
    assert "Failed: 0" in output


def test_csv_aggregation_uses_json_metrics_and_complexity(tmp_path, capsys):
    args = _args(tmp_path, ["jpeg_q22"])

    def fake_run(command, **kwargs):
        _write_success_result(command, model_name="jpeg_q22")
        return subprocess.CompletedProcess(command, 0)

    rows = run_benchmark.run_benchmarks(args, subprocess_runner=fake_run)
    assert rows[0]["parameters_m"] == 3.0
    assert rows[0]["gflops"] == 7.0
    assert rows[0]["time_ms"] == 11.0

    with (tmp_path / "output/benchmark.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 1
    assert csv_rows[0]["codec"] == "jpeg_q22"
    assert csv_rows[0]["bpp"] == "0.25"
    assert csv_rows[0]["ms_ssim"] == "0.96"
    assert csv_rows[0]["parameters_m"] == "3.0"
    output = capsys.readouterr().out
    assert "[1/1] START jpeg_q22 | env=illm" in output
    assert "[1/1] SUCCESS jpeg_q22 | elapsed=" in output
    assert "Completed: 1" in output
    assert "Success: 1" in output
    assert "Failed: 0" in output
