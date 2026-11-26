"""Audio -> text transcription using faster-whisper (free, local)."""
from __future__ import annotations

from .config import WHISPER_MODEL

_model = None


def _get_model():
    global _model
    if _model is None:
        # Imported lazily so the rest of the pipeline works even if
        # faster-whisper is not installed yet.
        from faster_whisper import WhisperModel
        # int8 runs comfortably on CPU (free Colab / any laptop).
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe(video_path: str) -> str:
    """Return the spoken text of a video as one string (empty if no speech)."""
    try:
        model = _get_model()
        segments, _info = model.transcribe(video_path, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:  # no audio, model missing, etc.
        print(f"[audio] transcription skipped for {video_path}: {exc}")
        return ""
