"""Frame extraction + perceptual hashing."""
from __future__ import annotations

import cv2
import imagehash
from PIL import Image

from .config import FRAMES_PER_SECOND


def extract_frames(video_path: str, fps: float = FRAMES_PER_SECOND):
    """Sample frames from a video and return them as a list of PIL images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(video_fps / fps)))

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    return frames


def extract_frame_hashes(video_path: str, fps: float = FRAMES_PER_SECOND):
    """Perceptual hash per sampled frame (robust to resize/crop/brightness)."""
    return [imagehash.phash(img) for img in extract_frames(video_path, fps)]
