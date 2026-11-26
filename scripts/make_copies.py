"""Generate edited 'copies' of your original videos to test the system.

Needs ffmpeg installed (free). Creates cropped, resized, flipped, brighter
and trimmed versions in data/copies/.

    python scripts/make_copies.py data/originals/myvideo.mp4
"""
import os
import subprocess
import sys

EDITS = {
    "crop":   ["-vf", "crop=in_w*0.8:in_h*0.8"],
    "resize": ["-vf", "scale=640:-2"],
    "flip":   ["-vf", "hflip"],
    "bright": ["-vf", "eq=brightness=0.15"],
    "trim":   ["-ss", "2", "-t", "15"],
}


def make_copies(src: str, out_dir: str = "data/copies"):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    for name, args in EDITS.items():
        out = os.path.join(out_dir, f"{stem}__{name}.mp4")
        cmd = ["ffmpeg", "-y", "-i", src, *args, out]
        print("->", out)
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/make_copies.py <original_video>")
    make_copies(sys.argv[1])
