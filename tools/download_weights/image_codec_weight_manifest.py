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
DEFAULT_LAB_SOURCE_ROOT = Path("/NEW_EDS/lisq")
OLD_LAB_SOURCE_ROOT = Path("/NEW_EDS/JJ_Group/lisq")


def _entry(
    target: str,
    *,
    legacy: str | None = None,
    local: list[str] | None = None,
    search_roots: list[str] | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    return {
        "target": target,
        "legacy": legacy,
        "local": local or [],
        "search_roots": search_roots or [],
        "url": url,
    }


LOCAL_OR_URL_CODEC_FILES: dict[str, list[dict[str, Any]]] = {
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
        _entry(
            "stablecodec/sd-turbo",
            legacy="/NEW_EDS/lisq/StableCodec/sd-turbo",
            local=["/NEW_EDS/lisq/StableCodec_Old/sd-turbo"],
        ),
        _entry(
            "stablecodec/elic_official.pth",
            legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/elic_official.pth",
            local=["/NEW_EDS/lisq/StableCodec/StableCodec/elic_official.pth"],
        ),
        _entry(
            "stablecodec/stablecodec_ft32.pkl",
            legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft32.pkl",
            local=["/NEW_EDS/lisq/StableCodec/StableCodec/stablecodec_ft32.pkl"],
        ),
        _entry(
            "stablecodec/stablecodec_ft16.pkl",
            legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft16.pkl",
            local=["/NEW_EDS/lisq/StableCodec/StableCodec/stablecodec_ft16.pkl"],
        ),
        _entry(
            "stablecodec/stablecodec_ft8.pkl",
            legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft8.pkl",
            local=["/NEW_EDS/lisq/StableCodec/StableCodec/stablecodec_ft8.pkl"],
        ),
        _entry(
            "stablecodec/stablecodec_ft4.pkl",
            legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft4.pkl",
            local=["/NEW_EDS/lisq/StableCodec/StableCodec/stablecodec_ft4.pkl"],
        ),
        _entry(
            "stablecodec/stablecodec_ft2.pkl",
            legacy="/NEW_EDS/JJ_Group/lisq/StableCodec_Old/StableCodec/stablecodec_ft2.pkl",
            local=["/NEW_EDS/lisq/StableCodec/StableCodec/stablecodec_ft2.pkl"],
        ),
    ],
    "tcm": [
        _entry("tcm/0.0025.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.0025.pth.tar"),
        _entry("tcm/0.0035.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.0035.pth.tar"),
        _entry("tcm/0.0067.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.0067.pth.tar"),
        _entry(
            "tcm/0.013.pth.tar",
            legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.013.pth.tar",
            local=["/NEW_EDS/lisq/LIC_TCM/0.013.pth..tar"],
        ),
        _entry("tcm/0.025.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.025.pth.tar"),
        _entry("tcm/0.05.pth.tar", legacy="/NEW_EDS/JJ_Group/lisq/LIC_TCM/0.05.pth.tar"),
    ],
    "cosmos": [
        _entry("cosmos/Cosmos-0.1-Tokenizer-DI16x16/encoder.jit", legacy="/NEW_EDS/lisq/Cosmos-Tokenizer/pretrained_ckpts/Cosmos-0.1-Tokenizer-DI16x16/encoder.jit"),
        _entry("cosmos/Cosmos-0.1-Tokenizer-DI16x16/decoder.jit", legacy="/NEW_EDS/lisq/Cosmos-Tokenizer/pretrained_ckpts/Cosmos-0.1-Tokenizer-DI16x16/decoder.jit"),
        _entry("cosmos/Cosmos-0.1-Tokenizer-DI8x8/encoder.jit", legacy="/NEW_EDS/lisq/Cosmos-Tokenizer/pretrained_ckpts/Cosmos-0.1-Tokenizer-DI8x8/encoder.jit"),
        _entry("cosmos/Cosmos-0.1-Tokenizer-DI8x8/decoder.jit", legacy="/NEW_EDS/lisq/Cosmos-Tokenizer/pretrained_ckpts/Cosmos-0.1-Tokenizer-DI8x8/decoder.jit"),
    ],
    "tatok": [
        _entry("tatok/ar_dtok_lp_256px.pth", search_roots=["/NEW_EDS/lisq/LlamaGen-main", "/NEW_EDS/lisq/pytorch-image-tokenizer-master", "/NEW_EDS/lisq/pytorch-image-tokenizer-master2", "/NEW_EDS/lisq/VAR"]),
        _entry("tatok/ta_tok.pth", search_roots=["/NEW_EDS/lisq/LlamaGen-main", "/NEW_EDS/lisq/pytorch-image-tokenizer-master", "/NEW_EDS/lisq/pytorch-image-tokenizer-master2", "/NEW_EDS/lisq/VAR"]),
        _entry("tatok/vq_ds16_t2i.pt", search_roots=["/NEW_EDS/lisq/LlamaGen-main", "/NEW_EDS/lisq/pytorch-image-tokenizer-master", "/NEW_EDS/lisq/pytorch-image-tokenizer-master2", "/NEW_EDS/lisq/VAR"]),
    ],
    "var": [
        _entry("var/vae_ch160v4096z32.pth", legacy="/NEW_EDS/lisq/VAR/vae_ch160v4096z32.pth"),
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
    parser = argparse.ArgumentParser(description="Download public image codec weights and stage private/local weights")
    choices = ["all", *all_codecs()]
    if selected_codec is None:
        parser.add_argument("--codec", choices=choices, default="all")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for all downloaded weights.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token, defaults to HF_TOKEN.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision.")
    parser.add_argument("--url-base", default=None, help="Optional base URL used for local/private files without hard-coded public URLs.")
    parser.add_argument("--source-root", default=None, help="Optional local root mirroring /NEW_EDS-style private weights.")
    parser.add_argument(
        "--lab-source-root",
        default=os.environ.get("CHANGAN_LAB_WEIGHT_SOURCE", str(DEFAULT_LAB_SOURCE_ROOT)),
        help="Lab-local source root used to resolve legacy /NEW_EDS paths.",
    )
    parser.add_argument("--skip-missing-private", action="store_true", help="Do not fail when private/local-only files cannot be found.")
    parser.add_argument("--strict-private", action="store_true", help="Fail when private/local-only files are missing, even for all-codec runs.")
    parser.add_argument("--public-only", action="store_true", help="Only download public Hugging Face/torch.hub weights; skip private/local staging.")
    parser.add_argument("--local-only", action="store_true", help="Only stage local files from /NEW_EDS-style paths; do not use network downloads.")
    parser.add_argument("--search-local", action="store_true", help="Search configured local directories for files that do not have exact source paths.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned local copies/downloads without writing files or using the network.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing target files.")
    args = parser.parse_args()

    codec = selected_codec or args.codec
    if args.public_only and args.strict_private:
        parser.error("--public-only and --strict-private cannot be used together.")
    if args.public_only and args.local_only:
        parser.error("--public-only and --local-only cannot be used together.")
    if args.skip_missing_private and args.strict_private:
        parser.error("--skip-missing-private and --strict-private cannot be used together.")

    if codec == "all" and not args.strict_private:
        args.skip_missing_private = True

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
    local_missing: list[str] = []
    if codec in LOCAL_OR_URL_CODEC_FILES and not args.public_only:
        local_missing = stage_local_or_url_files(
            LOCAL_OR_URL_CODEC_FILES[codec],
            output_root,
            source_root=Path(args.source_root).resolve() if args.source_root else None,
            lab_source_root=Path(args.lab_source_root).resolve(),
            url_base=args.url_base,
            force=args.force,
            dry_run=args.dry_run,
            allow_search=args.search_local,
        )

    if args.local_only:
        if codec not in LOCAL_OR_URL_CODEC_FILES:
            print(f"[{codec}] no local staging entries; skipped network-only downloads (--local-only).")
    elif codec in HF_FILE_GROUPS:
        for group in HF_FILE_GROUPS[codec]:
            download_hf_group(group, output_root, token=args.token, revision=args.revision, force=args.force, dry_run=args.dry_run)

    if not args.local_only and codec in TORCH_HUB_MODELS:
        download_torch_hub_group(TORCH_HUB_MODELS[codec], output_root, dry_run=args.dry_run)

    if codec in LOCAL_OR_URL_CODEC_FILES and args.public_only:
        print(
            f"[{codec}] skipped private/local-only files (--public-only). "
            "Provide --source-root, --lab-source-root, --url-base, or place them manually if this codec is needed."
        )

    if local_missing and args.skip_missing_private:
        missing_text = "\n    ".join(local_missing)
        print(
            f"[{codec}] local/private files were not staged. "
            "They are not publicly downloadable by this script unless an online source is configured:\n    " + missing_text
        )
    elif local_missing:
        missing_text = "\n    ".join(local_missing)
        raise FileNotFoundError(
            "local/private weight files are missing. Provide --source-root, "
            "--lab-source-root, --url-base, or place them manually:\n    " + missing_text
        )


def download_hf_group(group: dict[str, Any], output_root: Path, *, token: str | None, revision: str | None, force: bool, dry_run: bool) -> None:
    repo_id = group["repo_id"]
    target = output_root / group["target"]
    if dry_run:
        files = group.get("files")
        if files is None:
            print(f"would snapshot {repo_id} -> {target}")
        else:
            for filename in files:
                print(f"would download {repo_id}/{filename} -> {target / filename}")
        return

    target.mkdir(parents=True, exist_ok=True)
    files = group.get("files")

    if files is None:
        if target.exists() and any(target.iterdir()) and not force:
            print(f"exists {target}")
            return
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install huggingface_hub to download Hugging Face weights.") from exc
        print(f"snapshot {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            token=token,
            revision=revision,
        )
        return

    pending = [filename for filename in files if force or not (target / filename).exists()]
    if not pending:
        for filename in files:
            print(f"exists {target / filename}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install huggingface_hub to download Hugging Face weights.") from exc

    missing: list[str] = []
    for filename in pending:
        dst = target / filename
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


def download_torch_hub_group(group: dict[str, Any], output_root: Path, *, dry_run: bool) -> None:
    torch_home = output_root / group["torch_home"]
    if dry_run:
        for model_name in group["models"]:
            print(f"would torch.hub {group['repo']}:{model_name} -> TORCH_HOME={torch_home}")
        return

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install torch to prefetch torch.hub weights.") from exc

    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    for model_name in group["models"]:
        print(f"torch.hub {group['repo']}:{model_name} -> TORCH_HOME={torch_home}")
        model = torch.hub.load(group["repo"], model_name)
        del model


def stage_local_or_url_files(
    entries: list[dict[str, Any]],
    output_root: Path,
    *,
    source_root: Path | None,
    lab_source_root: Path,
    url_base: str | None,
    force: bool,
    dry_run: bool,
    allow_search: bool,
) -> list[str]:
    missing: list[str] = []
    for item in entries:
        target = output_root / str(item["target"])
        if target.exists() and not force:
            print(f"exists {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)

        source = resolve_local_source(item, source_root, lab_source_root, allow_search=allow_search)
        if source and source.exists():
            if dry_run:
                print(f"would copy {source} -> {target}")
                continue
            try:
                copy_path(source, target)
                print(f"copied {source} -> {target}")
            except OSError as exc:
                missing.append(f"{target} (copy failed from {source}: {exc})")
            continue

        url = item.get("url") or make_url(url_base, str(item["target"]))
        if url:
            if dry_run:
                print(f"would download {url} -> {target}")
                continue
            print(f"download {url} -> {target}")
            urllib.request.urlretrieve(url, target)
            continue

        missing.append(str(target))
    return missing


def resolve_local_source(
    item: dict[str, Any],
    source_root: Path | None,
    lab_source_root: Path,
    *,
    allow_search: bool,
) -> Path | None:
    if source_root is not None:
        candidate = source_root / str(item["target"])
        if candidate.exists():
            return candidate

    for candidate in local_source_candidates(item, lab_source_root):
        if candidate.exists():
            return candidate

    if allow_search:
        found = search_local_source(item, lab_source_root)
        if found is not None:
            return found

    return None


def local_source_candidates(item: dict[str, Any], lab_source_root: Path) -> list[Path]:
    raw_paths = []
    legacy = item.get("legacy")
    if legacy:
        raw_paths.append(str(legacy))
    raw_paths.extend(str(path) for path in item.get("local", []))

    candidates: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        candidates.append(path)
        candidates.extend(remap_lab_path(path, lab_source_root))

    return dedupe_paths(candidates)


def remap_lab_path(path: Path, lab_source_root: Path) -> list[Path]:
    raw = str(path)
    old_prefix = str(OLD_LAB_SOURCE_ROOT) + os.sep
    default_prefix = str(DEFAULT_LAB_SOURCE_ROOT) + os.sep

    if raw.startswith(old_prefix):
        suffix = raw[len(old_prefix):]
        return [lab_source_root / suffix, DEFAULT_LAB_SOURCE_ROOT / suffix]
    if raw.startswith(default_prefix):
        suffix = raw[len(default_prefix):]
        return [lab_source_root / suffix, OLD_LAB_SOURCE_ROOT / suffix]
    return []


def search_local_source(item: dict[str, Any], lab_source_root: Path) -> Path | None:
    roots = item.get("search_roots", [])
    if not roots:
        return None

    target_name = Path(str(item["target"])).name
    for raw_root in roots:
        root_candidates = dedupe_paths([Path(str(raw_root)), *remap_lab_path(Path(str(raw_root)), lab_source_root)])
        for root in root_candidates:
            if not root.is_dir():
                continue
            found = find_file_limited(root, target_name)
            if found is not None:
                return found
    return None


def find_file_limited(root: Path, filename: str, max_depth: int = 8) -> Path | None:
    root = root.resolve()
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if len(current.parts) - root_depth >= max_depth:
            dirnames[:] = []
        if filename in filenames:
            return current / filename
    return None


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def make_url(url_base: str | None, target: str) -> str | None:
    if not url_base:
        return None
    return url_base.rstrip("/") + "/" + target.replace("\\", "/")


def copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        copy_tree_plain(source, target)
    else:
        shutil.copyfile(source, target)


def copy_tree_plain(source: Path, target: Path) -> None:
    """Copy files without preserving metadata, which can fail on mounted disks."""
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        dst = target / path.relative_to(source)
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)


if __name__ == "__main__":
    main()
