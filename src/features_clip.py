"""CLIP deep-learning frame embeddings (the 'real ML' matching method).

Loaded only when METHOD == "clip" in config, so the lightweight hash method
keeps working even if torch / open_clip are not installed.

Install for this mode:  pip install torch open_clip_torch
"""
from __future__ import annotations

import numpy as np

_model = None
_preprocess = None
_device = "cpu"


def _load():
    global _model, _preprocess, _device
    if _model is None:
        import open_clip
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        _model = _model.to(_device).eval()
    return _model, _preprocess, _device


def frame_embeddings(images) -> np.ndarray:
    """Return an (n_frames, dim) array of L2-normalised CLIP embeddings."""
    if not images:
        return np.zeros((0, 512), dtype="float32")
    import torch
    model, preprocess, device = _load()
    batch = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # normalise
    return feats.cpu().numpy().astype("float32")
