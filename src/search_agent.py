"""One-video agent.

Give it a single ORIGINAL video. It:
  1. transcribes the audio -> text,
  2. pulls keywords from that text (+ optional title),
  3. searches YouTube and Dailymotion by those keywords,
  4. downloads the candidate videos,
  5. compares each candidate's frames + text against the original,
  6. returns the ones that cross the copy threshold, with their links.

Honest limit: this finds copies that are discoverable by KEYWORD search.
A re-upload with a totally unrelated title can still be missed, and free
search has daily quotas. It does NOT scan all of a platform by video content
(no public service can).
"""
from __future__ import annotations

import json
import sys
import os
import subprocess
import urllib.parse
import urllib.request

from .audio import transcribe
from .pipeline import compare_two

DOWNLOAD_DIR = "data/downloads"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi")

try:
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:  # handled gracefully
    CountVectorizer = None


# ---------------------------------------------------------------- keywords ---
def extract_keywords(text: str, top_k: int = 6) -> list[str]:
    """Pick the most frequent meaningful words/phrases from the transcript."""
    text = (text or "").strip()
    if not text or CountVectorizer is None:
        return []
    try:
        vec = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                              max_features=1000)
        counts = vec.fit_transform([text]).toarray()[0]
        vocab = vec.get_feature_names_out()
        ranked = sorted(zip(vocab, counts), key=lambda p: p[1], reverse=True)
        return [w for w, c in ranked[:top_k] if c > 0]
    except ValueError:
        return []


# ------------------------------------------------------------------ search ---
def search_youtube(query: str, limit: int = 5) -> list[dict]:
    """Search YouTube with yt-dlp (no API key needed)."""
    results = []
    try:
        out = subprocess.run(
            [sys.executable, "-m", "yt_dlp", f"ytsearch{limit}:{query}", "--flat-playlist",
             "--print", "%(webpage_url)s|%(title)s"],
            capture_output=True, text=True, check=False).stdout
        for line in out.splitlines():
            if "|" in line:
                url, title = line.split("|", 1)
                results.append({"url": url.strip(), "title": title.strip(),
                                "source": "youtube"})
    except Exception as exc:
        print(f"[search] youtube failed: {exc}")
    return results


def search_dailymotion(query: str, limit: int = 5) -> list[dict]:
    """Search Dailymotion via its free public API (no key needed)."""
    results = []
    try:
        params = urllib.parse.urlencode(
            {"search": query, "limit": limit, "fields": "id,title,url"})
        api = f"https://api.dailymotion.com/videos?{params}"
        with urllib.request.urlopen(api, timeout=20) as resp:
            data = json.load(resp)
        for item in data.get("list", []):
            results.append({"url": item.get("url"), "title": item.get("title"),
                            "source": "dailymotion"})
    except Exception as exc:
        print(f"[search] dailymotion failed: {exc}")
    return results


# ---------------------------------------------------------------- download ---
def download(url: str, out_dir: str = DOWNLOAD_DIR):
    os.makedirs(out_dir, exist_ok=True)
    before = set(os.listdir(out_dir))
    subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "mp4/bestvideo+bestaudio/best",
         "-o", os.path.join(out_dir, "%(id)s.%(ext)s"), url],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new = [f for f in (set(os.listdir(out_dir)) - before)
           if f.lower().endswith(VIDEO_EXTS)]
    return os.path.join(out_dir, new[0]) if new else None


# ------------------------------------------------------------------- agent ---
def find_copies(original_path: str, title: str = "",
                per_source: int = 5) -> dict:
    """Full one-video workflow. Returns query, keywords and ranked results."""
    transcript = transcribe(original_path)
    keywords = extract_keywords(transcript)

    query_parts = []
    if title.strip():
        query_parts.append(title.strip())
    query_parts.extend(keywords)
    if not query_parts:  # no speech, no title -> fall back to filename
        query_parts = [os.path.splitext(os.path.basename(original_path))[0]]
    query = " ".join(query_parts)

    candidates = search_youtube(query, per_source) + \
        search_dailymotion(query, per_source)

    results = []
    for c in candidates:
        if not c.get("url"):
            continue
        path = download(c["url"])
        if not path:
            continue
        r = compare_two(original_path, path)
        r.update({"url": c["url"], "title": c["title"],
                  "source": c["source"], "file": os.path.basename(path)})
        results.append(r)

    results.sort(key=lambda x: x["overlap"], reverse=True)
    return {"query": query, "keywords": keywords, "results": results}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Find copies of one video online")
    ap.add_argument("--original", required=True, help="Path to your video")
    ap.add_argument("--title", default="", help="Optional title/keywords")
    ap.add_argument("--per-source", type=int, default=5)
    args = ap.parse_args()

    out = find_copies(args.original, args.title, args.per_source)
    print(f"\nSearch query used: {out['query']}\n")
    if not out["results"]:
        print("No candidates found/downloaded.")
    for h in out["results"]:
        print(f"[{h['verdict']:16s}] theirs {h['suspect_coverage']*100:3.0f}% "
              f"/ yours {h['original_coverage']*100:3.0f}%  {h['source']:11s} "
              f"{h['title'][:45]}\n        {h['url']}")
