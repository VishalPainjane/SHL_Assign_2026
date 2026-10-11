"""
Submission Integrity & Quality Gate Validator
=============================================
Validates format, distribution statistics, and silence gating of candidate submissions.
Completely standalone with zero dependencies on past submission CSV files.
"""

import os
import numpy as np
import pandas as pd


class SubmissionValidator:
    """
    Validates submission format, values, distribution statistics, and silence gating.
    """
    def __init__(self, expected_rows: int = 216):
        self.expected_rows = expected_rows

    def validate(self, csv_path: str) -> dict:
        """
        Runs comprehensive quality gate validation checks on submission file.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Submission file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Assertions
        assert "filename" in df.columns and "label" in df.columns, (
            "Missing required columns ['filename', 'label']"
        )
        assert len(df) == self.expected_rows, (
            f"Expected exactly {self.expected_rows} rows, got {len(df)}"
        )
        assert df["label"].isna().sum() == 0, (
            "Found NaN values in submission!"
        )
        assert not np.isinf(df["label"]).any(), (
            "Found Infinite values in submission!"
        )
        assert (df["label"] >= 0.0).all() and (df["label"] <= 5.0).all(), (
            "Scores outside valid [0.0, 5.0] range!"
        )

        # Silence check (audio_66.wav)
        silence_score = df.loc[df["filename"] == "audio_66.wav", "label"].values
        has_silence_clamped = len(silence_score) > 0 and float(silence_score[0]) == 0.0

        stats = {
            "path": csv_path,
            "row_count": len(df),
            "mean": float(df["label"].mean()),
            "std": float(df["label"].std()),
            "min": float(df["label"].min()),
            "max": float(df["label"].max()),
            "silence_clamped": bool(has_silence_clamped),
        }
        return stats
