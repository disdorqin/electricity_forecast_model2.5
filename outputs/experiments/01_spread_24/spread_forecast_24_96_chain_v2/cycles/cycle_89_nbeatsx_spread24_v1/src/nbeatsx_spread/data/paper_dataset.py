from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PaperSample:
    """Minimal generic paper-profile sample, deliberately separate from business data."""

    y_backcast: np.ndarray
    x_backcast: np.ndarray
    x_future: np.ndarray
    y_future: np.ndarray


class PaperDataset(Dataset):
    """H=24 dataset adapter; no DA-RT or business cutoff semantics are used here."""

    def __init__(self, samples: list[PaperSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = self.samples[index]
        if s.y_backcast.shape != (168,) or s.y_future.shape != (24,):
            raise ValueError("paper profile requires L=168 and H=24")
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in {
            "y_backcast": s.y_backcast, "x_backcast": s.x_backcast,
            "x_future": s.x_future, "y_future": s.y_future,
        }.items()}
