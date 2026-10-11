"""
openSMILE eGeMAPSv02 Feature Extractor
======================================
Extracts 88 clinical and acoustic functionals from speech:
- Pitch (F0) inflection and rising slopes (indicative of natural sentence boundaries).
- Formant bandwidths (F1-F3) and energy distribution.
- Jitter and shimmer (voice stability and hesitation dynamics).
- Harmonics-to-Noise Ratio (HNR) and unvoiced pause durations.
"""

import os
import numpy as np


def extract_egemaps_features(audio_path: str) -> np.ndarray:
    """
    Extracts 88-D eGeMAPSv02 feature vector from audio file using openSMILE.
    Falls back to zeros if openSMILE is unavailable or file does not exist.
    """
    if not os.path.exists(audio_path):
        return np.zeros(88, dtype=np.float32)

    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        feats = smile.process_file(audio_path)
        return feats.values.flatten().astype(np.float32)
    except Exception:
        return np.zeros(88, dtype=np.float32)
