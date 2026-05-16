# External Video Tools

This directory is a placeholder for local video tool builds. Do not commit large
FFmpeg binaries unless the project owner explicitly asks for an offline bundle.

`cab.codec.ffmpeg_video.FFmpegVideoCodec` looks for FFmpeg in this order:

1. `ffmpeg_path` from the config file.
2. `tools/ffmpeg/bin/ffmpeg` under this repository.
3. `ffmpeg` from `PATH`.

H.264 requires an FFmpeg build with `libx264`.
H.265 requires `libx265`.
H.266/VVC requires `libvvenc` and VVC decode support.
