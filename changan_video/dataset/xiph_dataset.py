from __future__ import annotations

import urllib.request
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from ..h264_writer import H264Writer, VideoWriteConfig, VideoWriteStats


@dataclass(frozen=True)
class XiphSample:
    url: str
    width: int
    height: int
    fps: Fraction
    frames: int | None = None


@dataclass(frozen=True)
class VideoSourceInfo:
    width: int
    height: int
    fps: Fraction
    frames: int | None = None
    duration_seconds: float | None = None


_BASE = "https://media.xiph.org/video/derf/y4m"


def _s(name: str, w: int, h: int, fps: int | tuple, frames: int | None = None) -> XiphSample:
    f = Fraction(*fps) if isinstance(fps, tuple) else Fraction(fps, 1)
    return XiphSample(url=f"{_BASE}/{name}.y4m", width=w, height=h, fps=f, frames=frames)


XIPH_SAMPLES: dict[str, XiphSample] = {
    # ── QCIF 176×144 ─────────────────────────────────────────────────────────
    # MS-SSIM will be NaN for all QCIF sequences (both dims < 256).
    "bus_qcif_15fps":         _s("bus_qcif_15fps",         176, 144, 15,  75),
    "bus_qcif_7.5fps":        _s("bus_qcif_7.5fps",        176, 144, (15, 2), 38),
    "akiyo_qcif":             _s("akiyo_qcif",             176, 144, 30, 300),
    "foreman_qcif":           _s("foreman_qcif",           176, 144, 30, 300),
    "mobile_qcif":            _s("mobile_qcif",            176, 144, 30, 300),
    "coastguard_qcif":        _s("coastguard_qcif",        176, 144, 30, 300),
    "news_qcif":              _s("news_qcif",              176, 144, 30, 300),
    "silent_qcif":            _s("silent_qcif",            176, 144, 30, 300),
    "container_qcif":         _s("container_qcif",         176, 144, 30, 300),
    "mother_daughter_qcif":   _s("mother_daughter_qcif",   176, 144, 30, 300),
    "hall_monitor_qcif":      _s("hall_monitor_qcif",      176, 144, 30, 300),
    "football_qcif_15fps":    _s("football_qcif_15fps",    176, 144, 15, 130),
    "carphone_qcif":          _s("carphone_qcif",          176, 144, 30, 382),
    "suzie_qcif":             _s("suzie_qcif",             176, 144, 30, 150),
    "claire_qcif":            _s("claire_qcif",            176, 144, 30, 494),
    "grandma_qcif":           _s("grandma_qcif",           176, 144, 30, 870),
    "paris_qcif":             _s("paris_qcif",             176, 144, 30, 1065),
    "city_qcif_15fps":        _s("city_qcif_15fps",        176, 144, 15, 150),
    "crew_qcif_15fps":        _s("crew_qcif_15fps",        176, 144, 15, 150),
    "harbour_qcif_15fps":     _s("harbour_qcif_15fps",     176, 144, 15, 150),
    "ice_qcif_15fps":         _s("ice_qcif_15fps",         176, 144, 15, 120),
    "soccer_qcif_15fps":      _s("soccer_qcif_15fps",      176, 144, 15, 150),

    # ── CIF 352×288 ──────────────────────────────────────────────────────────
    "akiyo_cif":              _s("akiyo_cif",              352, 288, 30, 300),
    "bus_cif":                _s("bus_cif",                352, 288, 30, 150),
    "foreman_cif":            _s("foreman_cif",            352, 288, 30, 300),
    "mobile_cif":             _s("mobile_cif",             352, 288, 30, 300),
    "coastguard_cif":         _s("coastguard_cif",         352, 288, 30, 300),
    "football_cif":           _s("football_cif",           352, 288, 30, 260),
    "news_cif":               _s("news_cif",               352, 288, 30, 300),
    "silent_cif":             _s("silent_cif",             352, 288, 30, 300),
    "tempete_cif":            _s("tempete_cif",            352, 288, 30, 260),
    "flower_cif":             _s("flower_cif",             352, 288, 30, 250),
    "waterfall_cif":          _s("waterfall_cif",          352, 288, 30, 260),
    "mother_daughter_cif":    _s("mother_daughter_cif",    352, 288, 30, 300),
    "hall_monitor_cif":       _s("hall_monitor_cif",       352, 288, 30, 300),
    "container_cif":          _s("container_cif",          352, 288, 30, 300),
    "harbour_cif":            _s("harbour_cif",            352, 288, 30, 300),
    "city_cif":               _s("city_cif",               352, 288, 30, 300),
    "crew_cif":               _s("crew_cif",               352, 288, 30, 300),
    "soccer_cif":             _s("soccer_cif",             352, 288, 30, 300),
    "ice_cif":                _s("ice_cif",                352, 288, 30, 240),
    "paris_cif":              _s("paris_cif",              352, 288, 30, 1065),
    "deadline_cif":           _s("deadline_cif",           352, 288, 30, 1374),
    "highway_cif":            _s("highway_cif",            352, 288, 30, 2000),

    # ── 4CIF 704×576 (30 fps variants) ───────────────────────────────────────
    "city_4cif_30fps":        _s("city_4cif_30fps",        704, 576, 30, 300),
    "harbour_4cif_30fps":     _s("harbour_4cif_30fps",     704, 576, 30, 300),
    "crew_4cif_30fps":        _s("crew_4cif_30fps",        704, 576, 30, 300),
    "soccer_4cif_30fps":      _s("soccer_4cif_30fps",      704, 576, 30, 300),
    "ice_4cif_30fps":         _s("ice_4cif_30fps",         704, 576, 30, 240),

    # ── 720p 1280×720 ─────────────────────────────────────────────────────────
    "FourPeople_1280x720_60":    _s("FourPeople_1280x720_60",    1280, 720, 60, 600),
    "Johnny_1280x720_60":        _s("Johnny_1280x720_60",        1280, 720, 60, 600),
    "KristenAndSara_1280x720_60":_s("KristenAndSara_1280x720_60",1280, 720, 60, 600),
    "vidyo1_720p_60fps":         _s("vidyo1_720p_60fps",         1280, 720, 60, 600),
    "vidyo3_720p_60fps":         _s("vidyo3_720p_60fps",         1280, 720, 60, 600),
    "vidyo4_720p_60fps":         _s("vidyo4_720p_60fps",         1280, 720, 60, 600),
    "720p50_mobcal_ter":         _s("720p50_mobcal_ter",         1280, 720, 50, 504),
    "720p50_parkrun_ter":        _s("720p50_parkrun_ter",        1280, 720, 50, 504),
    "720p50_shields_ter":        _s("720p50_shields_ter",        1280, 720, 50, 504),
    "720p5994_stockholm_ter":    _s("720p5994_stockholm_ter",    1280, 720, (60000, 1001), 604),

    # ── 1080p 1920×1080 ───────────────────────────────────────────────────────
    "aspen_1080p":               _s("aspen_1080p",               1920, 1080, 30, 570),
    "blue_sky_1080p25":          _s("blue_sky_1080p25",          1920, 1080, 25, 217),
    "controlled_burn_1080p":     _s("controlled_burn_1080p",     1920, 1080, 30, 570),
    "crowd_run_1080p50":         _s("crowd_run_1080p50",         1920, 1080, 50, 500),
    "dinner_1080p30":            _s("dinner_1080p30",            1920, 1080, 30, 950),
    "ducks_take_off_1080p50":    _s("ducks_take_off_1080p50",    1920, 1080, 50, 500),
    "factory_1080p30":           _s("factory_1080p30",           1920, 1080, 30, 1339),
    "in_to_tree_1080p50":        _s("in_to_tree_1080p50",        1920, 1080, 50, 500),
    "life_1080p30":              _s("life_1080p30",              1920, 1080, 30, 825),
    "old_town_cross_1080p50":    _s("old_town_cross_1080p50",    1920, 1080, 50, 500),
    "park_joy_1080p50":          _s("park_joy_1080p50",          1920, 1080, 50, 500),
    "pedestrian_area_1080p25":   _s("pedestrian_area_1080p25",   1920, 1080, 25, 375),
    "red_kayak_1080p":           _s("red_kayak_1080p",           1920, 1080, 30, 570),
    "riverbed_1080p25":          _s("riverbed_1080p25",          1920, 1080, 25, 250),
    "rush_field_cuts_1080p":     _s("rush_field_cuts_1080p",     1920, 1080, 30, 570),
    "rush_hour_1080p25":         _s("rush_hour_1080p25",         1920, 1080, 25, 500),
    "snow_mnt_1080p":            _s("snow_mnt_1080p",            1920, 1080, 30, 570),
    "speed_bag_1080p":           _s("speed_bag_1080p",           1920, 1080, 30, 570),
    "station2_1080p25":          _s("station2_1080p25",          1920, 1080, 25, 313),
    "sunflower_1080p25":         _s("sunflower_1080p25",         1920, 1080, 25, 500),
    "touchdown_pass_1080p":      _s("touchdown_pass_1080p",      1920, 1080, 30, 570),
    "tractor_1080p25":           _s("tractor_1080p25",           1920, 1080, 25, 690),
    "west_wind_easy_1080p":      _s("west_wind_easy_1080p",      1920, 1080, 30, 570),
}


def _pick_video_rate(stream: av.video.stream.VideoStream) -> Fraction:
    rate = stream.average_rate or stream.base_rate or stream.guessed_rate
    if rate is None:
        warnings.warn(
            "Could not determine video frame rate; assuming 30 fps",
            RuntimeWarning,
            stacklevel=2,
        )
        return Fraction(30, 1)
    return Fraction(rate)


def probe_video_source(source: str | Path) -> VideoSourceInfo:
    """Open a video/Y4M source and return dimensions, fps, and frame count."""

    container = av.open(str(source), mode="r")
    try:
        stream = container.streams.video[0]
        fps = _pick_video_rate(stream)
        frames = stream.frames if stream.frames else None
        duration_seconds = None
        if frames is not None:
            duration_seconds = float(Fraction(frames, 1) / fps)

        return VideoSourceInfo(
            width=stream.width,
            height=stream.height,
            fps=fps,
            frames=frames,
            duration_seconds=duration_seconds,
        )
    finally:
        container.close()


def iter_video_frames(
    source: str | Path,
    limit: int | None = None,
    ndarray_format: str = "rgb24",
) -> Iterator[np.ndarray]:
    """Yield decoded frames from a video/Y4M source as uint8 numpy arrays."""

    container = av.open(str(source), mode="r")
    decoded = 0
    try:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format=ndarray_format)
            decoded += 1
            if limit is not None and decoded >= limit:
                break
    finally:
        container.close()


def download_sample(
    sample: XiphSample | str,
    dest_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download a Xiph Y4M sample to dest_dir and return the local path.

    Pass either a key from XIPH_SAMPLES or the XiphSample object directly.
    Skips the download if the file already exists, unless overwrite=True.
    """
    if isinstance(sample, str):
        sample = XIPH_SAMPLES[sample]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = sample.url.rsplit("/", 1)[-1]
    dest = dest_dir / filename
    if dest.exists() and not overwrite:
        return dest
    print(f"Downloading {filename} ...", flush=True)
    urllib.request.urlretrieve(sample.url, dest)
    return dest


def download_samples(
    samples: list[str] | None,
    dest_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download multiple samples; pass None to download all of XIPH_SAMPLES."""
    keys = samples if samples is not None else list(XIPH_SAMPLES)
    return {key: download_sample(key, dest_dir, overwrite=overwrite) for key in keys}


def transcode_video_source(
    source: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    codec: str = "libx264",
    crf: int | None = 23,
    preset: str | None = "veryfast",
) -> VideoWriteStats:
    """Decode a video/Y4M source and encode it as H.264."""

    container = av.open(str(source), mode="r")
    try:
        input_stream = container.streams.video[0]
        fps = _pick_video_rate(input_stream)

        config = VideoWriteConfig(
            output_path=output_path,
            width=input_stream.width,
            height=input_stream.height,
            fps=fps,
            codec=codec,
            crf=crf,
            preset=preset,
        )

        decoded = 0
        with H264Writer(config) as writer:
            for frame in container.decode(video=0):
                writer.write(frame)
                decoded += 1
                if limit is not None and decoded >= limit:
                    break
        return writer.stats
    finally:
        container.close()
