from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SequencePairDataset(Dataset):
    """
    Dataset of paired sequences and a continuous label (e.g., affinity).
    """

    def __init__(
        self,
        X_drug: np.ndarray,
        X_target: np.ndarray,
        y: np.ndarray,
    ):
        assert len(X_drug) == len(X_target) == len(y)
        self.X_drug = X_drug
        self.X_target = X_target
        self.y = y.astype("float32")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        drug = torch.tensor(self.X_drug[idx], dtype=torch.long)
        target = torch.tensor(self.X_target[idx], dtype=torch.long)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return drug, target, label


def generate_synthetic_data(
    n_samples: int = 1000,
    drug_seq_len: int = 100,
    target_seq_len: int = 250,
    drug_vocab_size: int = 30,
    target_vocab_size: int = 26,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random integer-encoded sequences and a synthetic regression target.

    This is only for demonstration; real projects would load
    actual tokenized biological sequences and experimental labels.
    """
    rng = np.random.default_rng(seed=42)

    X_drug = rng.integers(
        low=1, high=drug_vocab_size, size=(n_samples, drug_seq_len), dtype=np.int32
    )
    X_target = rng.integers(
        low=1, high=target_vocab_size, size=(n_samples, target_seq_len), dtype=np.int32
    )

    # create a synthetic signal: sum of first 10 tokens of each sequence + noise
    signal = (
        X_drug[:, :10].sum(axis=1)
        + X_target[:, :10].sum(axis=1) * 0.5
    )
    noise = rng.normal(loc=0.0, scale=5.0, size=n_samples)
    y = signal + noise

    # normalize somewhat
    y = (y - y.mean()) / y.std()

    return X_drug, X_target, y
