"""End-to-end comparison.

Two ways to use it:
1) Compare one original against one suspect video.
2) Compare one sample against every video in a reference folder and
   return a ranked list of likely copies (with links if available).
"""
from __future__ import annotations

import argparse
import json
import os

from .audio import transcribe
from .config import COPY_THRESHOLD, METHOD
from .frames import extract_frame_hashes, extract_frames
from .similarity import (clip_frame_similarity, combine, frame_similarity,
                         text_similarity)


def _frame_score(original_path: str, suspect_path: str) -> float:
    """Pick the frame-matching method from config (hash baseline or CLIP ML)."""
    if METHOD == "clip":
        from .features_clip import frame_embeddings
        emb_o = frame_embeddings(extract_frames(original_path))
        emb_s = frame_embeddings(extract_frames(suspect_path))
        return clip_frame_similarity(emb_o, emb_s)
    orig_hashes = extract_frame_hashes(original_path)
    suspect_hashes = extract_frame_hashes(suspect_path)
    return frame_similarity(orig_hashes, suspect_hashes)


def compare_two(original_path: str, suspect_path: str) -> dict:
    frame_score = _frame_score(original_path, suspect_path)

    orig_text = transcribe(original_path)
    suspect_text = transcribe(suspect_path)
    has_text = bool(orig_text) and bool(suspect_text)
    text_score = text_similarity(orig_text, suspect_text) if has_text else 0.0

    combined = combine(frame_score, text_score, has_text)
    return {
        "original": os.path.basename(original_path),
        "suspect": os.path.basename(suspect_path),
        "frame_score": round(frame_score, 3),
        "text_score": round(text_score, 3),
        "combined": round(combined, 3),
        "is_copy": combined >= COPY_THRESHOLD,
    }


def _load_links(reference_dir: str) -> dict:
    """Optional data/links.json maps filename -> source URL."""
    path = os.path.join(reference_dir, "links.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def search_collection(sample_path: str, reference_dir: str) -> list[dict]:
    """Rank every reference video by similarity to the sample."""
    links = _load_links(reference_dir)
    results = []
    for name in sorted(os.listdir(reference_dir)):
        if not name.lower().endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
            continue
        ref_path = os.path.join(reference_dir, name)
        r = compare_two(sample_path, ref_path)
        r["link"] = links.get(name, "")
        results.append(r)
    results.sort(key=lambda x: x["combined"], reverse=True)
    return results


def main():
    ap = argparse.ArgumentParser(description="Video copyright detection")
    ap.add_argument("--original", help="Path to the original/sample video")
    ap.add_argument("--suspect", help="Path to one suspected copy")
    ap.add_argument("--collection", help="Folder of reference videos to search")
    args = ap.parse_args()

    if args.suspect:
        print(json.dumps(compare_two(args.original, args.suspect), indent=2))
    elif args.collection:
        hits = search_collection(args.original, args.collection)
        for h in hits:
            tag = "COPY" if h["is_copy"] else "ok"
            link = f"  -> {h['link']}" if h.get("link") else ""
            print(f"[{tag}] {h['suspect']:40s} {h['combined']:.2f}{link}")
    else:
        ap.error("Provide --suspect OR --collection")


if __name__ == "__main__":
    main()
