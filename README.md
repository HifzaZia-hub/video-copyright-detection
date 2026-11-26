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

# Compare two videos directly (original vs suspected copy):
python -m src.pipeline --original data/originals/myvideo.mp4 --suspect data/downloads/suspect.mp4

# Or launch the website (works as website + mobile browser + desktop):
streamlit run app/streamlit_app.py
```

## How the score works
- Frames are sampled at ~1 per second from both videos.
- Each frame gets a perceptual hash; a suspect frame "matches" an original frame when the hashes are close
  (robust to crop, resize, brightness, flip, light filters).
- `frame_score` = share of original frames that found a match.
- `text_score` = TF-IDF cosine similarity of the two transcripts.
- `combined = 0.6 * frame_score + 0.4 * text_score` (auto-reweights if a video has no speech).
- Verdict = **COPY** when `combined >= THRESHOLD` (default 0.75, set in `src/config.py`).

## Group
Zainab Tariq (BSDSF24M004), Saneela Aziz (BSDSF24M022), Hifza Zia (BSDSF24M034) — BS Data Science, Machine Learning course.
