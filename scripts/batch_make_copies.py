"""Generate edited copies for EVERY video in data/originals/.

    python scripts/batch_make_copies.py

Needs ffmpeg installed (free). Output goes to data/copies/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_copies import make_copies  # noqa: E402

ORIG_DIR = "data/originals"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def main():
    if not os.path.isdir(ORIG_DIR):
        sys.exit(f"Folder not found: {ORIG_DIR}")
    videos = [f for f in os.listdir(ORIG_DIR) if f.lower().endswith(VIDEO_EXTS)]
    if not videos:
        sys.exit(f"No videos in {ORIG_DIR}. Put your originals there first.")

    for v in videos:
        print(f"\n== {v} ==")
        make_copies(os.path.join(ORIG_DIR, v))
    print(f"\nGenerated 5 edited copies for each of {len(videos)} original(s).")


if __name__ == "__main__":
    main()
