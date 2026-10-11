"""
WavLM-Large Feature Extractor
==============================
Extracts multi-stratum temporal representations from microsoft/wavlm-large (316M, 24 Layers).

Stratified Feature Design:
- Stratum 1 (Layers 4 & 8): Phonetic & articulatory tokens.
- Stratum 2 (Layers 12 & 16): Prosody, pacing, and speech cadence.
- Stratum 3 (Layers 20 & 24): High-level clause boundary and syntactic representations.
- Multi-stat temporal pooling: [mean, std, max, min] across time frames (4,096-D per stratum -> 12,288-D total).
"""

import os
import torch
import torchaudio
import numpy as np
from transformers import AutoFeatureExtractor, WavLMModel


def pool_frames(tensor_t_d: torch.Tensor) -> torch.Tensor:
    """Pools frame-level states over time dimension: [mean, std, max, min]."""
    mean_p = torch.mean(tensor_t_d, dim=0)
    std_p = torch.std(tensor_t_d, dim=0)
    max_p, _ = torch.max(tensor_t_d, dim=0)
    min_p, _ = torch.min(tensor_t_d, dim=0)
    return torch.cat([mean_p, std_p, max_p, min_p])  # 4096-D


def extract_wavlm_representations(
    audio_path: str,
    model: WavLMModel,
    feature_extractor: AutoFeatureExtractor,
    device: torch.device,
    max_seconds: int = 60,
) -> np.ndarray:
    """
    Extracts 12,288-D multi-stratum WavLM-Large representations for a single WAV file.
    """
    if not os.path.exists(audio_path):
        return np.zeros(12288, dtype=np.float32)

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Truncate to maximum 60 seconds
    waveform = waveform[:, :16000 * max_seconds].squeeze(0).numpy()
    if len(waveform) < 1600:
        return np.zeros(12288, dtype=np.float32)

    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Extract 3 strata
        s1 = 0.5 * (hidden_states[4][0] + hidden_states[8][0])
        s2 = 0.5 * (hidden_states[12][0] + hidden_states[16][0])
        s3 = 0.5 * (hidden_states[20][0] + hidden_states[24][0])

        p1 = pool_frames(s1)
        p2 = pool_frames(s2)
        p3 = pool_frames(s3)
        multi_stratum = torch.cat([p1, p2, p3]).cpu().numpy()

    return multi_stratum
