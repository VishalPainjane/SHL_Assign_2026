"""
Dataset Module
==============
Encapsulates PyTorch Dataset for tabular/embedding features
and maps targets to discrete ordinal indices for CORN loss training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from .models.corn import score_to_corn_label


class AcousticDataset(Dataset):
    """
    Dataset wrapping feature vectors and mapping continuous targets
    to CORN ordinal indices.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            # Map labels to discrete ordinal bins (0..10)
            corn_targets = score_to_corn_label(y)
            self.y = torch.tensor(corn_targets, dtype=torch.long)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        item = {"features": self.X[idx]}
        if self.y is not None:
            item["target"] = self.y[idx]
        return item
