from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from changan_video import H266WriteConfig, encode_frames_h266


def generated_frames(width: int, height: int, total: int):
    for i in range(total):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 50
        frame[:, :, 1] = 36
        frame[:, :, 2] = 100

        box = 48
        x = (i * 5) % (width - box)
        y = height // 2 - box // 2
        frame[y : y + box, x : x + box] = [225, 170, 64]
        yield frame


def main() -> None:
    width, height, fps, seconds = 320, 180, 30, 2
    output = Path("outputs/demo.h266")

    stats = encode_frames_h266(
        generated_frames(width, height, fps * seconds),
        H266WriteConfig(
            output_path=output,
            width=width,
            height=height,
            fps=fps,
            input_format="rgb24",
            qp=32,
        ),
        progress=lambda n: print(f"encoded {n} frames") if n % fps == 0 else None,
    )
    print(stats)


if __name__ == "__main__":
    main()
