"""Scoring: two-way frame coverage, text similarity, and the copy decision.

Key idea: measure COVERAGE in both directions instead of one averaged score.
- original_coverage = share of the ORIGINAL's frames found in the suspect
                      (how much of your video was used)
- suspect_coverage  = share of the SUSPECT's frames found in the original
                      (how much of their video is made from yours)

A trimmed 50% clip of your video -> original_coverage ~0.5, suspect_coverage
~1.0, so it is correctly seen as a (partial) copy and reported honestly.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (CLIP_SIM_THRESHOLD, HASH_MATCH_DISTANCE, TEXT_THRESHOLD,
                     VISUAL_THRESHOLD)


def hash_coverage(a_hashes, b_hashes) -> float:
    """Share of A's frames that have a close hash match somewhere in B. 0..1."""
    if not a_hashes or not b_hashes:
        return 0.0
    matched = 0
    for ah in a_hashes:
        best = min((ah - bh) for bh in b_hashes)  # hamming distance
        if best <= HASH_MATCH_DISTANCE:
            matched += 1
    return matched / len(a_hashes)


def clip_coverage(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Share of A's frames whose best cosine match in B is above threshold."""
    if emb_a is None or emb_b is None or len(emb_a) == 0 or len(emb_b) == 0:
        return 0.0
    sims = emb_a @ emb_b.T              # cosine (vectors are normalised)
    best = sims.max(axis=1)
    return float((best >= CLIP_SIM_THRESHOLD).mean())


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
        return 0.0


def decide(original_coverage: float, suspect_coverage: float,
           text_score: float) -> tuple[str, bool]:
    """Decide the verdict from the two coverages and the audio match.

    A strong VISUAL match alone is enough — different narration can never
    hide a visual copy. Audio only adds a flag, it never lowers the result.
    """
    orig_hi = original_coverage >= VISUAL_THRESHOLD
    susp_hi = suspect_coverage >= VISUAL_THRESHOLD

    if orig_hi and susp_hi:
        return "full copy", True
    if orig_hi or susp_hi:
        return "partial copy", True
    if text_score >= TEXT_THRESHOLD:
        return "audio/narration match", True
    return "not a copy", False
