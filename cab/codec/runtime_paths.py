from pathlib import Path


CODEC_RUNTIME_SUBDIRS = {
    "cab.codec.MLICPlusPlus.MLICPlusPlusImageCodec": "mlicplusplus",
    "cab.codec.hific.HiFiCImageCodec": "hific",
    "cab.codec.stablecodec.StableCodecImageCodec": "stablecodec",
}


def inject_codec_runtime_dir(config, codec_name, cache_dir):
    """Inject a writable runtime directory for codecs that create local files."""
    codec_config = config.get(codec_name)
    if not isinstance(codec_config, dict):
        raise KeyError(f"codec '{codec_name}' not found in config")

    codec_type = str(codec_config.get("type", ""))
    subdir = CODEC_RUNTIME_SUBDIRS.get(codec_type)
    if subdir is None:
        return config

    params = codec_config.setdefault("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"codec '{codec_name}' params must be a mapping")

    runtime_dir = Path(cache_dir).expanduser().resolve() / subdir
    params["runtime_dir"] = str(runtime_dir)
    return config


def _prepare_runtime_dir(runtime_dir):
    if runtime_dir is None:
        raise ValueError(
            "runtime_dir is required for this codec so runtime files do not "
            "write into the repository."
        )
    path = Path(runtime_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_mlicplusplus_stream_dir(runtime_dir, epoch):
    stream_dir = (
        _prepare_runtime_dir(runtime_dir)
        / "codestream"
        / f"{int(epoch) + 1:02d}"
    )
    stream_dir.mkdir(parents=True, exist_ok=True)
    return stream_dir


def prepare_hific_log_path(runtime_dir):
    return _prepare_runtime_dir(runtime_dir) / "hific.log"


def prepare_stablecodec_paths(runtime_dir):
    root = _prepare_runtime_dir(runtime_dir)
    reconstruction_dir = root / "reconstructions"
    bitstream_dir = root / "bitstreams"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    bitstream_dir.mkdir(parents=True, exist_ok=True)
    return reconstruction_dir, bitstream_dir
