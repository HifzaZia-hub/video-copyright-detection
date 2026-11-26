"""Scoring: frame similarity, text similarity, and their combination."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import FRAME_WEIGHT, HASH_MATCH_DISTANCE, TEXT_WEIGHT


def frame_similarity(orig_hashes, suspect_hashes) -> float:
    """Share of original frames that have a close match in the suspect video.

    Returns 0..1. Order-independent, so it tolerates trimming/reordering.
    """
    if not orig_hashes or not suspect_hashes:
        return 0.0
    matched = 0
    for oh in orig_hashes:
        best = min((oh - sh) for sh in suspect_hashes)  # hamming distance
        if best <= HASH_MATCH_DISTANCE:
            matched += 1
    return matched / len(orig_hashes)


def text_similarity(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity of two transcripts. Returns 0..1."""
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return 0.0
    try:
        tfidf = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        # happens if both texts are only stop-words
        return 0.0


def combine(frame_score: float, text_score: float, has_text: bool) -> float:
    """Weighted blend. If there is no usable speech, rely on frames only."""
    if not has_text:
        return frame_score
    return FRAME_WEIGHT * frame_score + TEXT_WEIGHT * text_score
