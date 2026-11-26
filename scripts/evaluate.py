"""Measure accuracy on your dataset — the numbers your report needs.

Reads data/manifest.csv (made by build_manifest.py). Every COPY should be
detected as a copy of its source ORIGINAL. We check every (copy, original)
pair and compute precision, recall, F1, plus a per-edit breakdown.

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
from src.config import COPY_THRESHOLD, METHOD  # noqa: E402
from src.pipeline import compare_two  # noqa: E402

MANIFEST = "data/manifest.csv"


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main():
    if not os.path.exists(MANIFEST):
        sys.exit("Run: python scripts/build_manifest.py first.")

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    originals = [r for r in rows if r["type"] == "original"]
    copies = [r for r in rows if r["type"] == "copy"]
    if not originals or not copies:
        sys.exit("Need both originals and copies in the manifest.")

    tp = fp = fn = tn = 0
    per_edit = defaultdict(lambda: {"hit": 0, "total": 0})

    print(f"Method = {METHOD}, threshold = {COPY_THRESHOLD}")
    print(f"Comparing {len(copies)} copies x {len(originals)} originals ...")

    for c in copies:
        truth = c["source_video"]
        for o in originals:
            is_true = (stem(o["path"]) == truth)
            score = compare_two(o["path"], c["path"])["combined"]
            pred = score >= COPY_THRESHOLD
            if pred and is_true:
                tp += 1
            elif pred and not is_true:
                fp += 1
            elif not pred and is_true:
                fn += 1
            else:
                tn += 1
            if is_true:
                per_edit[c["edit"]]["total"] += 1
                per_edit[c["edit"]]["hit"] += int(pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    print("\n===== RESULTS =====")
    print(f"Precision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F1-score  : {f1:.2f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("\nPer-edit recall (did it catch each edit type?):")
    for edit, d in sorted(per_edit.items()):
        r = d["hit"] / d["total"] if d["total"] else 0.0
        print(f"  {edit:8s} {r*100:5.1f}%  ({d['hit']}/{d['total']})")


if __name__ == "__main__":
    main()
