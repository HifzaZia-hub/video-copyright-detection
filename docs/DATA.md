# Part 2 — Building the dataset (free & legal)

Goal: gather **100–200 short videos**, then automatically make edited "copies"
of some of them so we can test whether the system detects copies.

## 1. Where to get videos (all free, legal)

Use **Creative Commons** or **public-domain** videos only — never random
copyrighted clips. Good free sources:

- **YouTube (Creative Commons filter):** search a topic → *Filters* → under
  *Features* pick **Creative Commons** → copy the links.
- **Pexels Videos / Pixabay Videos:** free stock clips, no account needed.
- **Internet Archive (archive.org):** large public-domain video collection.
- **VCDB dataset:** a research benchmark for partial video-copy detection.
  Search "VCDB partial copy detection dataset" — it is meant for academic use
  and may require a short access request to the authors. Optional; the sources
  above are enough to start.

Keep clips short (30s–2min) so processing stays fast on free Colab/Kaggle.

## 2. Folder layout

```
data/
├── originals/     <- your "real" videos (the ones you want to protect)
├── copies/        <- auto-generated edited versions (made by the scripts)
├── downloads/     <- videos downloaded from URLs (candidates / references)
├── sources.txt    <- list of video URLs to download (one per line)
└── links.json     <- filename -> source URL (auto-made; lets the app show links)
```

## 3. Workflow (the commands)

```bash
# A) Download reference videos listed in data/sources.txt
python scripts/download_cc.py

# B) Put a few of your own clips in data/originals/, then make edited copies
python scripts/batch_make_copies.py        # crop, resize, flip, brightness, trim

# C) Build the ground-truth table of the whole dataset
python scripts/build_manifest.py           # creates data/manifest.csv

# D) Test the engine on a copy (should score as a COPY):
python -m src.pipeline \
  --original data/originals/myclip.mp4 \
  --suspect  data/copies/myclip__crop.mp4
```

## 4. What each edit tests

`make_copies.py` creates five common tampering types per video:

| Edit | Simulates |
|------|-----------|
| crop | someone cropping out a watermark/edges |
| resize | re-uploading at a different resolution |
| flip | horizontal mirror to dodge simple matching |
| bright | brightness/filter change |
| trim | using only a 15-second section |

A good system should still recognise all of these as copies.

## 5. Legal note

This tool only measures **similarity**. It does not decide what is legally a
copyright violation — that is a human/legal judgement. Using only CC and
public-domain videos keeps your dataset free of any licensing cost or risk.
