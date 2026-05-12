# changan-video

Write H.264 from per-frame images retrieved from a database using PyAV, or use
[Xiph.org Test Media](https://media.xiph.org/) Y4M/video sources as encoding test inputs.

## Installation

```bash
pip install -r requirements.txt
```

For use as an importable package in another project, a development install is recommended:

```bash
pip install -e .
```

If your FFmpeg build does not include the `libx264` encoder, set `codec` in the config to
a H.264 encoder available on your machine, such as `h264_nvenc`, `h264_qsv`, or `libopenh264`.

## Write raw H.264

```python
from changan_video import VideoWriteConfig, encode_frames

stats = encode_frames(
    frames=my_frame_iterable,  # each frame is an HxWx3 uint8 RGB ndarray
    config=VideoWriteConfig(
        output_path="outputs/video.h264",
        width=1920,
        height=1080,
        fps=30,
        input_format="rgb24",
    ),
)
print(stats)
```

Raw `.h264` files usually require an explicit frame rate when playing back:

```bash
ffplay -framerate 30 outputs/video.h264
```

## Recommended: write H.264 inside an MP4 container

```python
from changan_video import VideoWriteConfig, encode_frames

stats = encode_frames(
    frames=my_frame_iterable,
    config=VideoWriteConfig(
        output_path="outputs/video.mp4",
        width=1920,
        height=1080,
        fps=30,
        input_format="rgb24",
    ),
)
```

## Connect your database

The database layer only needs to yield records; the encoding layer pulls frames from them:

```python
from changan_video import VideoWriteConfig, encode_records

def query_video_frames(video_id):
    # replace with your real database cursor/query
    # yield {"frame_rgb": np.ndarray(...)}
    ...

stats = encode_records(
    records=query_video_frames("video-001"),
    frame_getter=lambda row: row["frame_rgb"],
    config=VideoWriteConfig(
        output_path="outputs/video-001.mp4",
        width=1920,
        height=1080,
        fps=30,
        input_format="rgb24",
    ),
)
```

If your database stores frames in OpenCV-style BGR, set `input_format` to:

```python
input_format="bgr24"
```

## Test with the Xiph dataset

A small Xiph Derf sample is pre-configured in the package:

```text
https://media.xiph.org/video/derf/y4m/bus_qcif_15fps.y4m
```

Encode the first 75 frames from the command line:

```bash
python examples/encode_xiph_url.py --output outputs/xiph-bus.mp4 --limit 75
```

Or call it directly from code:

```python
from changan_video import XIPH_SAMPLES, transcode_video_source

sample = XIPH_SAMPLES["bus_qcif_15fps"]
stats = transcode_video_source(
    source=sample.url,
    output_path="outputs/xiph-bus.h264",
    limit=sample.frames,
)
```

If you have already downloaded the Xiph dataset locally, pass a local `.y4m` path as `source`.

## Scripts

Run the complete smoke test on Windows PowerShell:

```powershell
.\scripts\run_smoke.ps1 -InstallDeps
```

Run it without the network-backed Xiph sample:

```powershell
.\scripts\run_smoke.ps1 -SkipXiph
```

If PowerShell blocks local scripts because of execution policy, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_smoke.ps1 -SkipXiph
```

Run the Xiph sample with more frames:

```powershell
.\scripts\run_smoke.ps1 -XiphLimit 75
```

On Linux/macOS or GitHub Actions:

```bash
bash scripts/run_smoke.sh --install-deps --xiph-limit 5
```

## Demo

Generate a test raw H.264 file:

```bash
python examples/demo_generate_h264.py
```

Generate an MP4 simulating a database-driven workflow:

```bash
python examples/db_usage_example.py
```

## Key points

- Each frame must be `uint8`, typically shaped `height x width x 3`.
- `VideoWriteConfig.width/height` must match the input frames; strict checking is on by default.
- Always use `close()` or `with H264Writer(...)`; the H.264 encoder buffers trailing frames that must be flushed.
- An `output_path` ending in `.h264` writes a raw bitstream; `.mp4` writes an MP4 container.
