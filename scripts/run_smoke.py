from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print(flush=True)
    print("$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def ensure_dependencies(install_deps: bool) -> None:
    missing = []
    for module_name in ("av", "numpy"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if not missing:
        import av

        print(f"PyAV: {av.__version__}", flush=True)
        return

    if not install_deps:
        modules = ", ".join(missing)
        raise SystemExit(
            f"Missing dependencies: {modules}. "
            "Run again with --install-deps, or install requirements.txt first."
        )

    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    for module_name in missing:
        importlib.import_module(module_name)


def assert_output(path: str) -> None:
    output = ROOT / path
    if not output.exists() or output.stat().st_size <= 0:
        raise SystemExit(f"Expected non-empty output file: {output}")
    print(f"OK: {path} ({output.stat().st_size} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local H.264 smoke tests.")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install requirements.txt before running tests if dependencies are missing.",
    )
    parser.add_argument(
        "--skip-xiph",
        action="store_true",
        help="Skip the network-backed Xiph sample test.",
    )
    parser.add_argument(
        "--xiph-limit",
        type=int,
        default=5,
        help="Number of Xiph sample frames to encode.",
    )
    args = parser.parse_args()

    ensure_dependencies(args.install_deps)

    run([sys.executable, "-m", "compileall", "changan_video", "examples"])

    run([sys.executable, "examples/demo_generate_h264.py"])
    assert_output("outputs/demo.h264")

    run([sys.executable, "examples/db_usage_example.py"])
    assert_output("outputs/video-001.mp4")

    if not args.skip_xiph:
        run(
            [
                sys.executable,
                "examples/encode_xiph_url.py",
                "--output",
                "outputs/xiph-bus-smoke.mp4",
                "--limit",
                str(args.xiph_limit),
            ]
        )
        assert_output("outputs/xiph-bus-smoke.mp4")

    print(flush=True)
    print("Smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
