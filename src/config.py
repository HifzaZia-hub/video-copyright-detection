"""Central settings. Change numbers here to tune the system."""

# Matching method for video frames:
#   "hash" -> perceptual hashing (fast, free, no GPU)  [baseline]
#   "clip" -> CLIP deep-learning embeddings (real ML, better on hard edits)
# For "clip":  pip install -r requirements-clip.txt
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

# A video is flagged as a (partial) copy when this share of EITHER video's
# frames is found in the other. 0.70 = 70%. This is a *coverage* threshold,
# not an average, so a 50% partial copy is still reported honestly as 50%.
VISUAL_THRESHOLD = 0.70

# Separately flag an audio/narration match at or above this text similarity.
# (Audio can only ADD a flag; it never lowers a visual match.)
TEXT_THRESHOLD = 0.60

# Whisper model size for audio->text: tiny / base / small / medium.
# "base" is a good free balance; "tiny" is fastest.
WHISPER_MODEL = "base"
