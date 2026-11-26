# Video Copyright Detection (Free / Open-Source)

Detect whether a video is a copy of an *original* video by comparing:
1. **Audio → text** (transcript similarity), and
2. **Video frames** (perceptual-hash / visual similarity).

If the combined similarity is **≥ 70–80%**, the system flags it as a likely copy
and shows the matching video (and its link, if the reference was downloaded from a URL).

> **Scope (important):** This searches *your own reference collection* of videos, **not the whole internet**.
> Platforms like YouTube do **not** allow searching by video *content* — only by keywords/title.
> So "find my copy anywhere online" is done in two realistic ways:
> - Compare the sample against a limited reference collection you build, **or**
> - (Optional) search YouTube by **keywords** via the free YouTube Data API, download candidates, then rank them.

This matches the Limitations already stated in the project proposal.

## 100% free stack
| Need | Free tool |
|------|-----------|
| Code | Python |
| Compute / GPU | Google Colab or Kaggle (free) |
| Audio → text | `faster-whisper` (runs locally, free) |
| Frame similarity | `imagehash` (perceptual hash) — optional CLIP upgrade |
| Text similarity | `scikit-learn` TF-IDF — optional sentence-transformers upgrade |
| Fast search | `faiss` (optional, for large collections) |
| Website | `streamlit`, hosted free on Streamlit Community Cloud or Hugging Face Spaces |
| Desktop app | same app run locally, or packaged with `PyInstaller` (free) |
| Mobile app | the web app as a **PWA** (free); native Flutter is optional |
| Downloading candidates | `yt-dlp` (free) |
| Repo | GitHub (free) |

## Project parts (roadmap)
- **Part 1 — Setup + core engine + website**  ← included here
- **Part 2 — Data:** collect 100–200 clips (VCDB + Creative Commons) and auto-make edited copies (`scripts/make_copies.py`)
- **Part 3 — Accuracy upgrade:** add CLIP frame embeddings + FAISS; tune the 0.70–0.80 threshold
- **Part 4 — Desktop app:** package the same app with PyInstaller
- **Part 5 — Mobile app:** turn the web app into a PWA (installable on phone)
- **Part 6 — Optional YouTube keyword search** (free API quota)
- **Part 7 — Evaluation + report:** precision, recall, F1, mAP vs a hashing baseline

## Quick start
```bash
pip install -r requirements.txt
# For the ML method, also: pip install -r requirements-clip.txt
#   then set METHOD = "clip" in src/config.py

# One-video agent: upload nothing — point it at your video, it searches
# YouTube + Dailymotion by keywords, downloads candidates, and returns matches:
python -m src.search_agent --original data/originals/myvideo.mp4

# Compare two videos directly (original vs suspected copy):
python -m src.pipeline --original data/originals/myvideo.mp4 --suspect data/downloads/suspect.mp4

# Or launch the website (works as website + mobile browser + desktop):
streamlit run app/streamlit_app.py
```

## One-video online search (closest to "find my copy")
Mode **"Find copies online (upload one)"** in the website: you upload a single
video, the agent transcribes it, extracts keywords, searches **YouTube** and
**Dailymotion**, downloads the candidate videos, compares each against yours,
and lists the ones above the threshold **with their links**.

**Honest limit:** this finds copies that are *discoverable by keyword search*.
A re-upload with a totally unrelated title can be missed, and free search has
daily quotas. No public tool can scan an entire platform by video *content* —
only the platforms themselves (e.g. YouTube Content ID) can, on private
infrastructure. This keyword approach is the realistic free version.

## How the score works (full vs partial copies)
- Frames are sampled at ~1 per second from both videos and matched
  (perceptual hash by default; CLIP embeddings in ML mode). Robust to crop,
  resize, brightness, flip and light filters.
- We measure **coverage in both directions** instead of one averaged number:
  - `original_coverage` = share of **your** frames found in the suspect
    (how much of your video was used)
  - `suspect_coverage` = share of **their** frames found in your original
    (how much of their video is made from yours)
- `text_score` = TF-IDF similarity of the transcripts (audio).
- **Verdict** (threshold `VISUAL_THRESHOLD`, default 0.70, in `src/config.py`):
  - **full copy** — both coverages high
  - **partial copy** — one coverage high (e.g. a trimmed 50% clip: their
    coverage ~100%, your coverage ~50% → reported honestly as partial)
  - **audio/narration match** — visuals differ but transcripts match
  - a strong **visual** match alone is enough — different narration can never
    hide a visual copy (audio only adds a flag, never lowers the result)

## Group
Zainab Tariq (BSDSF24M004), Saneela Aziz (BSDSF24M022), Hifza Zia (BSDSF24M034) — BS Data Science, Machine Learning course.
