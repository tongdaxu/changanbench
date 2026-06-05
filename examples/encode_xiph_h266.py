from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from changan_video import XIPH_SAMPLES, transcode_video_source_h266


def main() -> None:
    default_sample = XIPH_SAMPLES["bus_qcif_15fps"]

    parser = argparse.ArgumentParser(description="Encode a Xiph Y4M/video source to H.266.")
    parser.add_argument("--source", default=default_sample.url, help="Local path or URL.")
    parser.add_argument("--output", default="outputs/xiph-bus-h266.h266", help="Output .h266/.vvc or supported container.")
    parser.add_argument("--limit", type=int, default=75, help="Max frames to encode.")
    parser.add_argument("--qp", type=int, default=32)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()

    stats = transcode_video_source_h266(
        source=args.source,
        output_path=args.output,
        limit=args.limit,
        qp=args.qp,
        preset=args.preset,
    )
    print(stats)


if __name__ == "__main__":
    main()
