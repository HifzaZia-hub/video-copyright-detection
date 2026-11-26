"""Website / desktop / mobile UI (one Streamlit app covers all three).

Run locally:   streamlit run app/streamlit_app.py
Host for free: push to GitHub, then deploy on Streamlit Community Cloud
               or Hugging Face Spaces. The same URL opens on a phone browser
               and can be 'Added to Home Screen' as a mobile app (PWA).
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import COPY_THRESHOLD  # noqa: E402
from src.pipeline import compare_two, search_collection  # noqa: E402
from src.search_agent import find_copies  # noqa: E402

st.set_page_config(page_title="Video Copyright Detector", page_icon="🎬")
st.title("🎬 Video Copyright Detector")
st.caption("Compare audio text + video frames to find likely copies. 100% free & open-source.")


def _save_upload(upload) -> str:
    suffix = os.path.splitext(upload.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.read())
    tmp.close()
    return tmp.name


mode = st.radio(
    "Mode",
    ["Find copies online (upload one)", "Compare two videos", "Search my collection"],
)

# --------------------------------------------------- one-video online agent ---
if mode == "Find copies online (upload one)":
    st.write("Upload **one** video. The agent pulls keywords from its audio, "
             "searches YouTube + Dailymotion, downloads candidates, and checks "
             "each one against your video.")
    original = st.file_uploader("Your original video", type=["mp4", "mkv", "webm", "mov"])
    title = st.text_input("Optional: title / extra keywords", "")
    per_source = st.slider("Candidates per site", 3, 10, 5)

    st.caption("⚠️ Finds copies discoverable by keyword search — not every copy "
               "on the internet. Free search has daily limits.")

    if st.button("Find copies", disabled=not original):
        with st.spinner("Searching, downloading and comparing… this can take a while."):
            out = find_copies(_save_upload(original), title=title, per_source=per_source)
        st.write(f"**Search query used:** {out['query']}")
        if not out["results"]:
            st.info("No candidates found or downloaded. Try adding a title/keywords.")
        for h in out["results"]:
            is_copy = h["combined"] >= COPY_THRESHOLD
            head = "⚠️ COPY" if is_copy else "ok"
            st.markdown(
                f"**{h['combined']*100:.0f}%** — {head} — *{h['source']}*  \n"
                f"[{h['title']}]({h['url']})  \n"
                f"frames {h['frame_score']*100:.0f}% · text {h['text_score']*100:.0f}% "
                f"· saved as `{h['file']}`")

# ----------------------------------------------------------- compare two ---
elif mode == "Compare two videos":
    c1, c2 = st.columns(2)
    original = c1.file_uploader("Original video", type=["mp4", "mkv", "webm", "mov"])
    suspect = c2.file_uploader("Suspected copy", type=["mp4", "mkv", "webm", "mov"])

    if st.button("Check", disabled=not (original and suspect)):
        with st.spinner("Analysing…"):
            result = compare_two(_save_upload(original), _save_upload(suspect))
        st.metric("Combined similarity", f"{result['combined']*100:.0f}%")
        st.write(f"Frame match: **{result['frame_score']*100:.0f}%**  |  "
                 f"Text match: **{result['text_score']*100:.0f}%**")
        if result["is_copy"]:
            st.error(f"⚠️ Likely a COPY (≥ {int(COPY_THRESHOLD*100)}% threshold)")
        else:
            st.success("✅ Not a copy by the current threshold")

# -------------------------------------------------------- search folder ---
else:
    st.info("Put your reference videos in **data/downloads/** "
            "(optionally add data/downloads/links.json for source URLs).")
    sample = st.file_uploader("Your sample video", type=["mp4", "mkv", "webm", "mov"])
    folder = st.text_input("Reference folder", value="data/downloads")

    if st.button("Search", disabled=not sample):
        with st.spinner("Searching your collection…"):
            hits = search_collection(_save_upload(sample), folder)
        for h in hits:
            label = "⚠️ COPY" if h["is_copy"] else "ok"
            line = f"**{h['suspect']}** — {h['combined']*100:.0f}%  ({label})"
            if h.get("link"):
                line += f" — [source]({h['link']})"
            st.write(line)
