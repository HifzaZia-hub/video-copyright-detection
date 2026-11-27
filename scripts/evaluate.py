"""Measure accuracy on your dataset — the numbers your report needs.

Reads data/manifest.csv (made by build_manifest.py). Every COPY should be
detected as a copy of its source ORIGINAL. We check every (copy, original)
pair and compute precision, recall and F1 — reported separately for FULL
copies and PARTIAL copies.

    python scripts/build_manifest.py     # first, build the ground truth
    python scripts/evaluate.py           # then, measure

Switch METHOD between "hash" and "clip" in src/config.py and run twice to
compare baseline vs ML in your report.

Speed: each video's frames and transcript are computed ONCE and reused, so
runtime grows with the number of videos, not the number of pairs.
"""
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio import transcribe  # noqa: E402
from src.config import METHOD  # noqa: E402
from src.frames import extract_frame_hashes, extract_frames  # noqa: E402
from src.similarity import (clip_coverage, decide, hash_coverage,  # noqa: E402
                            text_similarity)

MANIFEST = "data/manifest.csv"
PARTIAL_EDITS = {"trim", "half"}   # these keep only part of the original

_cache = {}


def signature(path):
    """Compute (frame_signature, transcript) for a video once, then reuse."""
    if path not in _cache:
        if METHOD == "clip":
            from src.features_clip import frame_embeddings
            sig = frame_embeddings(extract_frames(path))
        else:
            sig = extract_frame_hashes(path)
        _cache[path] = (sig, transcribe(path))
    return _cache[path]


def coverages(a_sig, b_sig):
    """Return (coverage of A in B, coverage of B in A)."""
    if METHOD == "clip":
        return clip_coverage(a_sig, b_sig), clip_coverage(b_sig, a_sig)
    return hash_coverage(a_sig, b_sig), hash_coverage(b_sig, a_sig)


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    if not os.path.exists(MANIFEST):
        sys.exit("Run: python scripts/build_manifest.py first.")

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    originals = [r for r in rows if r["type"] == "original"]
    copies = [r for r in rows if r["type"] == "copy"]
    if not originals or not copies:
        sys.exit("Need both originals and copies in the manifest.")

    tp = fp = fn = tn = 0
    split = {"full": [0, 0], "partial": [0, 0]}   # [caught, total]
    per_edit = defaultdict(lambda: {"hit": 0, "total": 0})

    print(f"Method = {METHOD}")
    print(f"Preparing {len(originals) + len(copies)} videos "
          f"(frames + transcripts, once each)...")

    for c in copies:
        c_sig, c_text = signature(c["path"])
        kind = "partial" if c["edit"] in PARTIAL_EDITS else "full"
        for o in originals:
            o_sig, o_text = signature(o["path"])
            is_true = (stem(o["path"]) == c["source_video"])
            orig_cov, susp_cov = coverages(o_sig, c_sig)
            text = (text_similarity(o_text, c_text)
                    if (o_text and c_text) else 0.0)
            pred = decide(orig_cov, susp_cov, text)[1]
            if pred and is_true:
                tp += 1
            elif pred and not is_true:
                fp += 1
            elif not pred and is_true:
                fn += 1
            else:
                tn += 1
            if is_true:
                split[kind][1] += 1
                split[kind][0] += int(pred)
                per_edit[c["edit"]]["total"] += 1
                per_edit[c["edit"]]["hit"] += int(pred)

    p, r, f1 = prf(tp, fp, fn)
    print("\n===== RESULTS =====")
    print(f"Precision : {p:.2f}")
    print(f"Recall    : {r:.2f}")
    print(f"F1-score  : {f1:.2f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    print("\nRecall by copy type:")
    for kind in ("full", "partial"):
        hit, total = split[kind]
        print(f"  {kind:8s} {(hit / total * 100 if total else 0):5.1f}%  "
              f"({hit}/{total})")

    print("\nRecall by edit:")
    for edit, d in sorted(per_edit.items()):
        print(f"  {edit:8s} {(d['hit'] / d['total'] * 100 if d['total'] else 0):5.1f}%"
              f"  ({d['hit']}/{d['total']})")


if __name__ == "__main__":
    main()
