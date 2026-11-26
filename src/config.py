"""Central settings. Change numbers here to tune the system."""

# Matching method for video frames:
#   "hash" -> perceptual hashing (fast, free, no GPU)  [baseline]
#   "clip" -> CLIP deep-learning embeddings (real ML, better on hard edits)
# Run the evaluation with each one to compare baseline vs ML in your report.
METHOD = "hash"

# How many frames per second to sample from each video.
FRAMES_PER_SECOND = 1.0

# --- hash method ---
# Perceptual-hash distance (0-64). Two frames "match" when their hash
# difference is <= this. Lower = stricter.
HASH_MATCH_DISTANCE = 10

# --- clip method ---
# Cosine similarity (0-1) above which two frames count as the same shot.
CLIP_SIM_THRESHOLD = 0.90

# Weights for combining the two signals.
FRAME_WEIGHT = 0.6
TEXT_WEIGHT = 0.4

# Final decision threshold (0.70 - 0.80 as in the proposal).
COPY_THRESHOLD = 0.75

# Whisper model size for audio->text: tiny / base / small / medium.
# "base" is a good free balance; "tiny" is fastest.
WHISPER_MODEL = "base"
