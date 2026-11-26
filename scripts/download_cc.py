"""Download the videos listed in data/sources.txt (free, via yt-dlp) and
record each file's source URL in data/downloads/links.json so the app can
show the link of a matched copy.

    python scripts/download_cc.py

Only use Creative Commons / public-domain videos. Needs: pip install yt-dlp
"""
import json
import os
import subprocess
import sys

DOWNLOAD_DIR = "data/downloads"
SOURCES = "data/sources.txt"
IDMAP = os.path.join(DOWNLOAD_DIR, "_idmap.txt")
LINKS = os.path.join(DOWNLOAD_DIR, "links.json")


def read_urls(path: str) -> list[str]:
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    if not os.path.exists(SOURCES):
        sys.exit(f"Create {SOURCES} first (one video URL per line).")

    urls = read_urls(SOURCES)
    if not urls:
        sys.exit(f"No URLs found in {SOURCES} (add some non-# lines).")

    open(IDMAP, "w").close()  # reset the id map for this run

    cmd = [
        "yt-dlp",
        "-f", "mp4/bestvideo+bestaudio/best",
        "--ignore-errors",
        "-o", os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        # record "filename|source_url" for every downloaded video
        "--print-to-file", "%(id)s.%(ext)s|%(webpage_url)s", IDMAP,
        "-a", SOURCES,
    ]
    print(f"Downloading {len(urls)} video(s) into {DOWNLOAD_DIR} ...")
    subprocess.run(cmd, check=False)

    links = {}
    if os.path.exists(LINKS):
        with open(LINKS, encoding="utf-8") as f:
            links = json.load(f)
    if os.path.exists(IDMAP):
        with open(IDMAP, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    name, url = line.split("|", 1)
                    links[name] = url

    with open(LINKS, "w", encoding="utf-8") as f:
        json.dump(links, f, indent=2)
    print(f"Done. {len(links)} link(s) recorded in {LINKS}")


if __name__ == "__main__":
    main()
