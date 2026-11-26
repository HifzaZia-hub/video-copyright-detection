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
from .config import METHOD
from .frames import extract_frame_hashes, extract_frames
from .similarity import (clip_coverage, decide, hash_coverage, text_similarity)


def _frame_coverages(original_path: str, suspect_path: str):
    """Return (original_coverage, suspect_coverage) using the configured method.

    Frames are extracted once per video, then compared both ways.
    """
    if METHOD == "clip":
        from .features_clip import frame_embeddings
        emb_o = frame_embeddings(extract_frames(original_path))
        emb_s = frame_embeddings(extract_frames(suspect_path))
        return clip_coverage(emb_o, emb_s), clip_coverage(emb_s, emb_o)

    orig_hashes = extract_frame_hashes(original_path)
    suspect_hashes = extract_frame_hashes(suspect_path)
    return (hash_coverage(orig_hashes, suspect_hashes),
            hash_coverage(suspect_hashes, orig_hashes))


def compare_two(original_path: str, suspect_path: str) -> dict:
    orig_cov, susp_cov = _frame_coverages(original_path, suspect_path)

    orig_text = transcribe(original_path)
    suspect_text = transcribe(suspect_path)
    has_text = bool(orig_text) and bool(suspect_text)
    text_score = text_similarity(orig_text, suspect_text) if has_text else 0.0

    verdict, is_copy = decide(orig_cov, susp_cov, text_score)
    return {
        "original": os.path.basename(original_path),
        "suspect": os.path.basename(suspect_path),
        "original_coverage": round(orig_cov, 3),   # how much of YOURS was used
        "suspect_coverage": round(susp_cov, 3),     # how much of THEIRS is yours
        "overlap": round(max(orig_cov, susp_cov), 3),  # headline number
        "text_score": round(text_score, 3),
        "verdict": verdict,
        "is_copy": is_copy,
    }


def _load_links(reference_dir: str) -> dict:
    """Optional data/links.json maps filename -> source URL."""
    path = os.path.join(reference_dir, "links.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def search_collection(sample_path: str, reference_dir: str) -> list[dict]:
    """Rank every reference video by overlap with the sample."""
    links = _load_links(reference_dir)
    results = []
    for name in sorted(os.listdir(reference_dir)):
        if not name.lower().endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
            continue
        ref_path = os.path.join(reference_dir, name)
        r = compare_two(sample_path, ref_path)
        r["link"] = links.get(name, "")
        results.append(r)
    results.sort(key=lambda x: x["overlap"], reverse=True)
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
            link = f"  -> {h['link']}" if h.get("link") else ""
            print(f"[{h['verdict']:16s}] {h['suspect']:36s} "
                  f"yours {h['original_coverage']:.2f} / theirs "
                  f"{h['suspect_coverage']:.2f}{link}")
    else:
        ap.error("Provide --suspect OR --collection")


if __name__ == "__main__":
    main()
