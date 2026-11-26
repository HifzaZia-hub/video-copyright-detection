"""Frame extraction + perceptual hashing."""
from __future__ import annotations

import cv2
import imagehash
from PIL import Image

from .config import FRAMES_PER_SECOND


def extract_frame_hashes(video_path: str, fps: float = FRAMES_PER_SECOND):
    """Sample frames from a video and return a list of perceptual hashes.

    pHash is fairly robust to resize, crop, brightness and light filters,
    which are the common edits made to copied videos.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(video_fps / fps)))

    hashes = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            hashes.append(imagehash.phash(pil))
        idx += 1
    cap.release()
    return hashes
