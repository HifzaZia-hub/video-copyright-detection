"""Generate edited 'copies' of your original videos to test the system.

Needs ffmpeg + ffprobe installed (free). Creates cropped, resized, flipped,
brighter, trimmed and half-length versions in data/copies/.

    python scripts/make_copies.py data/originals/myvideo.mp4

Edit types:
  crop/resize/flip/bright  -> FULL copies (whole video, edited)
  trim (15s) / half (50%)  -> PARTIAL copies (only part of the original)
"""
import os
import subprocess
import sys

FULL_EDITS = {
    "crop":   ["-vf", "crop=in_w*0.8:in_h*0.8"],
    "resize": ["-vf", "scale=640:-2"],
    "flip":   ["-vf", "hflip"],
    "bright": ["-vf", "eq=brightness=0.15"],
}


def duration(src: str):
    """Video length in seconds via ffprobe, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", src],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None


def make_copies(src: str, out_dir: str = "data/copies"):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]

    edits = dict(FULL_EDITS)
    edits["trim"] = ["-ss", "2", "-t", "15"]          # a 15s partial
    dur = duration(src)
    if dur and dur > 4:
        edits["half"] = ["-ss", "0", "-t", f"{dur / 2:.2f}"]  # first 50%

    for name, args in edits.items():
        out = os.path.join(out_dir, f"{stem}__{name}.mp4")
        cmd = ["ffmpeg", "-y", "-i", src, *args, out]
        print("->", out)
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/make_copies.py <original_video>")
    make_copies(sys.argv[1])
