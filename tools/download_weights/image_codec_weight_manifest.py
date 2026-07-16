from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/data9-2/BenchmarkData/weights")


def _entry(target: str, *, legacy: str | None = None, url: str | None = None) -> dict[str, str | None]:
    return {"target": target, "legacy": legacy, "url": url}


LOCAL_OR_URL_CODEC_FILES: dict[str, list[dict[str, str | None]]] = {
    "bsq": [
        _entry("bsq/BSQ_Vit/checkpoint.pt", legacy="/NEW_EDS/JJ_Group/lisq/bsq-vit/BSQ_Vit/checkpoint.pt"),
        _entry("bsq/BSQ_Vit_18/checkpoint.pt", legacy="/NEW_EDS/JJ_Group/lisq/bsq-vit/BSQ_Vit_18/checkpoint.pt"),
    ],
    "diffeic": [
        _entry("diffeic/v2-1_512-ema-pruned.ckpt", legacy="/NEW_EDS/JJ_Group/lisq/DiffEIC/weight/v2-1_512-ema-pruned.ckpt"),
        _entry("diffeic/1_2_16/lc.ckpt", legacy="/NEW_EDS/JJ_Group/lisq/DiffEIC/weight/1_2_16/lc.ckpt"),
        _entry("diffeic/1_2_8/lc.ckpt", legacy="/NEW_EDS/JJ_Group/lisq/DiffEIC/weight/1_2_8/lc.ckpt"),
        _entry("diffeic/1_2_4/lc.ckpt", legacy="/NEW_EDS/JJ_Group/lisq/DiffEIC/weight/1_2_4/lc.ckpt"),
        _entry("diffeic/1_2_2/lc.ckpt", legacy="/NEW_EDS/JJ_Group/lisq/DiffEIC/weight/1_2_2/lc.ckpt"),
    ],
    "elic": [
        _entry("elic/ELIC_0004_ft_3980_Plateau.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/ELiC-ReImplemetation/ELIC_0004_ft_3980_Plateau.pth.tar"),
        _entry("elic/ELIC_0008_ft_3980_Plateau.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/ELiC-ReImplemetation/ELIC_0008_ft_3980_Plateau.pth.tar"),
        _entry("elic/ELIC_0016_ft_3980_Plateau.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/ELiC-ReImplemetation/ELIC_0016_ft_3980_Plateau.pth.tar"),
        _entry("elic/ELIC_0032_ft_3980_Plateau.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/ELiC-ReImplemetation/ELIC_0032_ft_3980_Plateau.pth.tar"),
        _entry("elic/ELIC_0150_ft_3980_Plateau.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/ELiC-ReImplemetation/ELIC_0150_ft_3980_Plateau.pth.tar"),
        _entry("elic/ELIC_0450_ft_3980_Plateau.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/ELiC-ReImplemetation/ELIC_0450_ft_3980_Plateau.pth.tar"),
    ],
    "flowmo": [
        _entry("flowmo/flowmo_lo.pth", legacy="/NEW_EDS/JJ_Group/lisq/FlowMo/flowmo_lo.pth"),
        _entry("flowmo/flowmo_hi.pth", legacy="/NEW_EDS/JJ_Group/lisq/FlowMo/flowmo_hi.pth"),
    ],
    "fsq": [
        _entry("fsq/fsq_q0/epoch=000015.ckpt", legacy="/NEW_EDS/lisq/pytorch-image-tokenizer-master2/logs/2025-08-11T11-22-58_-sd3unet_fsq_888555/checkpoints/epoch=000015.ckpt"),
        _entry("fsq/fsq_q1/epoch=000015.ckpt", legacy="/NEW_EDS/lisq/pytorch-image-tokenizer-master2/logs/2025-07-28T15-57-01_-sd3unet_fsq_888555/checkpoints/epoch=000015.ckpt"),
        _entry("fsq/fsq_q2/epoch=000003.ckpt", legacy="/NEW_EDS/lisq/pytorch-image-tokenizer-master2/logs/2025-07-26T10-04-00_-sd3unet_fsq_888555/checkpoints/epoch=000003.ckpt"),
    ],
    "hific": [
        _entry("hific/hific_low.pt", legacy="/NEW_EDS/JJ_Group/lisq/high-fidelity-generative-compression/hific_low.pt"),
        _entry("hific/hific_med.pt", legacy="/NEW_EDS/JJ_Group/lisq/high-fidelity-generative-compression/hific_med.pt"),
        _entry("hific/hific_hi.pt", legacy="/NEW_EDS/JJ_Group/lisq/high-fidelity-generative-compression/hific_hi.pt"),
    ],
    "ibq": [
        _entry("ibq/imagenet256_1024.ckpt", legacy="/NEW_EDS/JJ_Group/lisq/SEED-Voken/imagenet256_1024.ckpt"),
        _entry("ibq/imagenet256_8192.ckpt", legacy="/NEW_EDS/JJ_Group/lisq/SEED-Voken/imagenet256_8192.ckpt"),
        _entry("ibq/imagenet256_16384.ckpt", legacy="/NEW_EDS/JJ_Group/lisq/SEED-Voken/imagenet256_16384.ckpt"),
        _entry("ibq/imagenet256_262144.ckpt", legacy="/NEW_EDS/JJ_Group/lisq/SEED-Voken/imagenet256_262144.ckpt"),
    ],
    "infinity": [
        _entry("infinity/infinity_vae_d16.pth", legacy="/NEW_EDS/JJ_Group/lisq/Infinity/infinity_vae_d16.pth"),
        _entry("infinity/infinity_vae_d24.pth", legacy="/NEW_EDS/JJ_Group/lisq/Infinity/infinity_vae_d24.pth"),
        _entry("infinity/infinity_vae_d32.pth", legacy="/NEW_EDS/JJ_Group/lisq/Infinity/infinity_vae_d32.pth"),
        _entry("infinity/infinity_vae_d64.pth", legacy="/NEW_EDS/JJ_Group/lisq/Infinity/infinity_vae_d64.pth"),
    ],
    "mlicpp": [
        _entry("mlicpp/mlicpp_new_mse_q1.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/MLIC-main/MLIC++/mlicpp_new_mse_q1.pth.tar"),
        _entry("mlicpp/mlicpp_new_mse_q2.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/MLIC-main/MLIC++/mlicpp_new_mse_q2.pth.tar"),
        _entry("mlicpp/mlicpp_new_mse_q3.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/MLIC-main/MLIC++/mlicpp_new_mse_q3.pth.tar"),
        _entry("mlicpp/mlicpp_new_mse_q4.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/MLIC-main/MLIC++/mlicpp_new_mse_q4.pth.tar"),
        _entry("mlicpp/mlicpp_new_mse_q5.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/MLIC-main/MLIC++/mlicpp_new_mse_q5.pth.tar"),
        _entry("mlicpp/mlicpp_new_mse_q6.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/MLIC-main/MLIC++/mlicpp_new_mse_q6.pth.tar"),
    ],
    "perco": [
        _entry("perco/Perco_0.0019_model/cmvl_2024_full_train", legacy="/NEW_EDS/lisq/bsq-vit/Perco_0.0019_model/cmvl_2024_full_train"),
        _entry("perco/Perco_0.0313_model/cmvl_2024_full_train", legacy="/NEW_EDS/lisq/bsq-vit/Perco_0.0313_model/cmvl_2024_full_train"),
        _entry("perco/Perco_0.125_model/cmvl_2024_full_train", legacy="/NEW_EDS/lisq/bsq-vit/Perco_0.125_model/cmvl_2024_full_train"),
    ],
    "ssdd": [
        _entry("ssdd/F16C4/F16C4_M_256.safetensors", legacy="/NEW_EDS/lisq/SSDD/weight/models--facebook--SSDD/snapshots/a29742837196f405d5e8af289d66356d14258bb0/F16C4/F16C4_M_256.safetensors"),
        _entry("ssdd/F8C4/F8C4_M_256.safetensors", legacy="/NEW_EDS/lisq/SSDD/weight/models--facebook--SSDD/snapshots/a29742837196f405d5e8af289d66356d14258bb0/F8C4/F8C4_M_256.safetensors"),
    ],
    "stablecodec": [
        _entry("stablecodec/elic_official.pth", legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/elic_official.pth"),
        _entry("stablecodec/stablecodec_ft32.pkl", legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft32.pkl"),
        _entry("stablecodec/stablecodec_ft16.pkl", legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft16.pkl"),
        _entry("stablecodec/stablecodec_ft8.pkl", legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft8.pkl"),
        _entry("stablecodec/stablecodec_ft4.pkl", legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft4.pkl"),
        _entry("stablecodec/stablecodec_ft2.pkl", legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft2.pkl"),
    ],
    "tcm": [
        _entry("tcm/0.0025.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.0025.pth.tar"),
        _entry("tcm/0.0035.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.0035.pth.tar"),
        _entry("tcm/0.0067.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.0067.pth.tar"),
        _entry("tcm/0.013.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.013.pth.tar"),
        _entry("tcm/0.025.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.025.pth.tar"),
        _entry("tcm/0.05.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.05.pth.tar"),
    ],
}


HF_FILE_GROUPS: dict[str, list[dict[str, Any]]] = {
    "cosmos": [
        {"repo_id": "nvidia/Cosmos-0.1-Tokenizer-DI16x16", "target": "cosmos/Cosmos-0.1-Tokenizer-DI16x16", "files": ["encoder.jit", "decoder.jit"]},
        {"repo_id": "nvidia/Cosmos-0.1-Tokenizer-DI8x8", "target": "cosmos/Cosmos-0.1-Tokenizer-DI8x8", "files": ["encoder.jit", "decoder.jit"]},
    ],
    "tatok": [
        {"repo_id": "csuhan/TA-Tok", "target": "tatok", "files": ["ar_dtok_lp_256px.pth", "ta_tok.pth"]},
        {"repo_id": "peizesun/llamagen_t2i", "target": "tatok", "files": ["vq_ds16_t2i.pt"]},
    ],
    "var": [
        {"repo_id": "FoundationVision/var", "target": "var", "files": ["vae_ch160v4096z32.pth"]},
    ],
    "ssdd": [
        {"repo_id": "facebook/SSDD", "target": "ssdd", "files": ["F16C4/F16C4_M_256.safetensors", "F8C4/F8C4_M_256.safetensors"], "optional": True},
    ],
    "stablecodec": [
        {"repo_id": "stabilityai/sd-turbo", "target": "stablecodec/sd-turbo", "files": None},
    ],
}


TORCH_HUB_MODELS: dict[str, dict[str, Any]] = {
    "msillm": {
        "repo": "facebookresearch/NeuralCompression",
        "models": [
            "msillm_quality_vlo1",
            "msillm_quality_vlo2",
            "msillm_quality_1",
            "msillm_quality_2",
            "msillm_quality_3",
            "msillm_quality_4",
            "msillm_quality_5",
            "msillm_quality_6",
        ],
        "torch_home": "msillm/torch_home",
    }
}


def public_codecs() -> list[str]:
    return sorted(set(HF_FILE_GROUPS) | set(TORCH_HUB_MODELS))


def local_or_url_codecs() -> list[str]:
    return sorted(LOCAL_OR_URL_CODEC_FILES)


def all_codecs() -> list[str]:
    return sorted(set(public_codecs()) | set(local_or_url_codecs()))


def main(selected_codec: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download or stage image codec weights")
    choices = ["all", *all_codecs()]
    if selected_codec is None:
        parser.add_argument("--codec", choices=choices, default="all")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for all downloaded weights.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token, defaults to HF_TOKEN.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision.")
    parser.add_argument("--url-base", default=None, help="Optional base URL used for local/private files without hard-coded public URLs.")
    parser.add_argument("--source-root", default=None, help="Optional local root mirroring /NEW_EDS-style private weights.")
    parser.add_argument("--skip-missing-private", action="store_true", help="Do not fail when private/local-only files cannot be found.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing target files.")
    args = parser.parse_args()

    codec = selected_codec or args.codec
    codecs = all_codecs() if codec == "all" else [codec]
    output_root = Path(args.output_root).resolve()

    failures: list[str] = []
    for name in codecs:
        try:
            download_codec(name, output_root, args)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
            print(f"[{name}] failed: {exc}", file=sys.stderr)

    if failures:
        print("\nSome weight groups were not completed:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(1)


def download_codec(codec: str, output_root: Path, args: argparse.Namespace) -> None:
    print(f"\n== {codec} ==")
    if codec in HF_FILE_GROUPS:
        for group in HF_FILE_GROUPS[codec]:
            download_hf_group(group, output_root, token=args.token, revision=args.revision, force=args.force)
    if codec in TORCH_HUB_MODELS:
        download_torch_hub_group(TORCH_HUB_MODELS[codec], output_root)
    if codec in LOCAL_OR_URL_CODEC_FILES:
        missing = stage_local_or_url_files(
            LOCAL_OR_URL_CODEC_FILES[codec],
            output_root,
            source_root=Path(args.source_root).resolve() if args.source_root else None,
            url_base=args.url_base,
            force=args.force,
        )
        if missing and not args.skip_missing_private:
            missing_text = "\n    ".join(missing)
            raise FileNotFoundError(
                "private/local weight files are missing. Provide --source-root, "
                "--url-base, or place them manually:\n    " + missing_text
            )


def download_hf_group(group: dict[str, Any], output_root: Path, *, token: str | None, revision: str | None, force: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install huggingface_hub to download Hugging Face weights.") from exc

    repo_id = group["repo_id"]
    target = output_root / group["target"]
    target.mkdir(parents=True, exist_ok=True)
    files = group.get("files")

    if files is None:
        print(f"snapshot {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            token=token,
            revision=revision,
        )
        return

    missing: list[str] = []
    for filename in files:
        dst = target / filename
        if dst.exists() and not force:
            print(f"exists {dst}")
            continue
        try:
            src = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                revision=revision,
                local_dir=str(target),
                local_dir_use_symlinks=False,
            )
            print(f"downloaded {repo_id}/{filename} -> {src}")
        except Exception:
            if group.get("optional"):
                missing.append(filename)
                continue
            raise

    if missing:
        print(
            f"optional files not found in {repo_id}: {', '.join(missing)}. "
            "Use --source-root or place them manually if your config references them."
        )


def download_torch_hub_group(group: dict[str, Any], output_root: Path) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install torch to prefetch torch.hub weights.") from exc

    torch_home = output_root / group["torch_home"]
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    for model_name in group["models"]:
        print(f"torch.hub {group['repo']}:{model_name} -> TORCH_HOME={torch_home}")
        model = torch.hub.load(group["repo"], model_name)
        del model


def stage_local_or_url_files(
    entries: list[dict[str, str | None]],
    output_root: Path,
    *,
    source_root: Path | None,
    url_base: str | None,
    force: bool,
) -> list[str]:
    missing: list[str] = []
    for item in entries:
        target = output_root / str(item["target"])
        if target.exists() and not force:
            print(f"exists {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)

        source = resolve_local_source(item, source_root)
        if source and source.exists():
            copy_path(source, target)
            print(f"copied {source} -> {target}")
            continue

        url = item.get("url") or make_url(url_base, str(item["target"]))
        if url:
            print(f"download {url} -> {target}")
            urllib.request.urlretrieve(url, target)
            continue

        missing.append(str(target))
    return missing


def resolve_local_source(item: dict[str, str | None], source_root: Path | None) -> Path | None:
    legacy = item.get("legacy")
    if source_root is not None:
        return source_root / str(item["target"])
    if legacy:
        return Path(legacy)
    return None


def make_url(url_base: str | None, target: str) -> str | None:
    if not url_base:
        return None
    return url_base.rstrip("/") + "/" + target.replace("\\", "/")


def copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


if __name__ == "__main__":
    main()
