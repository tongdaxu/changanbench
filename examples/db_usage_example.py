from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from changan_video import VideoWriteConfig, encode_records


def query_video_frames_from_db(video_id: str):
    """Replace this generator with your real database cursor/query."""

    width, height, total = 640, 360, 90
    for index in range(total):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = index % 255
        frame[:, :, 1] = 120
        frame[:, :, 2] = 220
        yield {
            "video_id": video_id,
            "frame_index": index,
            "frame_rgb": frame,
        }


def main() -> None:
    video_id = "video-001"

    stats = encode_records(
        records=query_video_frames_from_db(video_id),
        frame_getter=lambda row: row["frame_rgb"],
        config=VideoWriteConfig(
            output_path=f"outputs/{video_id}.mp4",
            width=640,
            height=360,
            fps=30,
            input_format="rgb24",
        ),
    )
    print(stats)


if __name__ == "__main__":
    main()
