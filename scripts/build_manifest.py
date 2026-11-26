"""Build data/manifest.csv — a table of every video in the dataset.

For copies, the originating video and the edit type are parsed from the
filename pattern  <original>__<edit>.mp4  produced by make_copies.py.
This manifest is the ground truth used later for evaluation (Part 7).

    python scripts/build_manifest.py
"""
import csv
import os

ORIG_DIR = "data/originals"
COPY_DIR = "data/copies"
OUT = "data/manifest.csv"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def is_video(name: str) -> bool:
    return name.lower().endswith(VIDEO_EXTS)


def main():
    rows = [["path", "type", "source_video", "edit"]]

    for f in sorted(os.listdir(ORIG_DIR)) if os.path.isdir(ORIG_DIR) else []:
        if is_video(f):
            rows.append([os.path.join(ORIG_DIR, f), "original", "", ""])

    for f in sorted(os.listdir(COPY_DIR)) if os.path.isdir(COPY_DIR) else []:
        if not is_video(f):
            continue
        stem = os.path.splitext(f)[0]
        if "__" in stem:
            source, edit = stem.rsplit("__", 1)
        else:
            source, edit = "", "unknown"
        rows.append([os.path.join(COPY_DIR, f), "copy", source, edit])

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {len(rows) - 1} entries to {OUT}")


if __name__ == "__main__":
    main()
