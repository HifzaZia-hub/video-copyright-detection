"""Measure accuracy on your dataset — the numbers your report needs.

Reads data/manifest.csv (made by build_manifest.py). Every COPY should be
detected as a copy of its source ORIGINAL. We check every (copy, original)
pair, use the system's own verdict (is_copy), and compute precision, recall
and F1 — reported separately for FULL copies and PARTIAL copies.

    python scripts/build_manifest.py     # first, build the ground truth
    python scripts/evaluate.py           # then, measure

Switch METHOD between "hash" and "clip" in src/config.py and run twice to
compare baseline vs ML in your report.

Note: this compares all copies against all originals, so keep the test set
modest (e.g. 10-20 originals) or it gets slow.
"""
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import METHOD  # noqa: E402
from src.pipeline import compare_two  # noqa: E402

MANIFEST = "data/manifest.csv"
PARTIAL_EDITS = {"trim", "half"}   # these keep only part of the original


def stem(path: str) -> str:
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

    # overall + split counters
    tp = fp = fn = tn = 0
    split = {"full": [0, 0], "partial": [0, 0]}  # [caught, total] true matches
    per_edit = defaultdict(lambda: {"hit": 0, "total": 0})

    print(f"Method = {METHOD}")
    print(f"Comparing {len(copies)} copies x {len(originals)} originals ...")

    for c in copies:
        truth = c["source_video"]
        kind = "partial" if c["edit"] in PARTIAL_EDITS else "full"
        for o in originals:
            is_true = (stem(o["path"]) == truth)
            pred = compare_two(o["path"], c["path"])["is_copy"]
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
        rec = hit / total if total else 0.0
        print(f"  {kind:8s} {rec*100:5.1f}%  ({hit}/{total})")

    print("\nRecall by edit:")
    for edit, d in sorted(per_edit.items()):
        rec = d["hit"] / d["total"] if d["total"] else 0.0
        print(f"  {edit:8s} {rec*100:5.1f}%  ({d['hit']}/{d['total']})")


if __name__ == "__main__":
    main()
