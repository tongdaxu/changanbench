from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from changan_video import VideoWriteConfig, encode_frames


def generated_frames(width: int, height: int, total: int):
    for i in range(total):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 26
        frame[:, :, 1] = 54
        frame[:, :, 2] = 92

        box = 72
        x = (i * 8) % (width - box)
        y = height // 2 - box // 2
        frame[y : y + box, x : x + box] = [235, 86, 47]
        yield frame


def main() -> None:
    width, height, fps, seconds = 640, 360, 30, 4
    output = Path("outputs/demo.h264")

    stats = encode_frames(
        generated_frames(width, height, fps * seconds),
        VideoWriteConfig(
            output_path=output,
            width=width,
            height=height,
            fps=fps,
            input_format="rgb24",
        ),
        progress=lambda n: print(f"encoded {n} frames") if n % fps == 0 else None,
    )
    print(stats)


if __name__ == "__main__":
    main()
