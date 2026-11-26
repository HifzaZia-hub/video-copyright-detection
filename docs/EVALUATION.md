# Measuring accuracy + baseline vs ML (for the report)

This is the part an ML course grades most: *numbers* that show how well the
system works, and a comparison between a simple baseline and a real ML method.

## The two methods

In `src/config.py` set `METHOD`:

- `METHOD = "hash"` — **baseline**: perceptual hashing. Fast, free, no GPU.
- `METHOD = "clip"` — **ML**: CLIP deep-learning frame embeddings. Needs
  `pip install torch open_clip_torch` (runs free on Google Colab's GPU).

## Steps

```bash
# 1. Prepare data (Part 2)
python scripts/batch_make_copies.py    # makes edited copies of your originals
python scripts/build_manifest.py       # writes data/manifest.csv (ground truth)

# 2. Evaluate the BASELINE
#    (set METHOD = "hash" in src/config.py)
python scripts/evaluate.py

# 3. Evaluate the ML method
#    (set METHOD = "clip" in src/config.py)
python scripts/evaluate.py
```

## What you get

```
Precision : 0.93
Recall    : 0.88
F1-score  : 0.90
TP=..  FP=..  FN=..  TN=..

Per-edit recall:
  bright    100.0%
  crop       80.0%
  flip       60.0%
  resize    100.0%
  trim       90.0%
```

## How to read it (plain words)

- **Precision** — of the videos it flagged as copies, how many really were.
- **Recall** — of the real copies, how many it managed to catch.
- **F1** — one overall score balancing the two (higher = better).
- **Per-edit recall** — which edit types it handles well (e.g. it may catch
  `crop` easily but struggle with `flip`). Great material for the report.

## For the report / viva

Put both runs side by side in a small table:

| Method | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Hash (baseline) | … | … | … |
| CLIP (ML) | … | … | … |

Then discuss: which edits each method handles, where CLIP helps, and the
trade-off (CLIP is more accurate but needs more compute). That discussion is
exactly what demonstrates the machine-learning understanding.
