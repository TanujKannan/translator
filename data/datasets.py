from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """
    Thin wrapper around parallel data. Expects pre-encoded token id sequences
    (already containing special tokens like <sos>/<eos> if you use them).
    """

    def __init__(self, src_sequences: Sequence[Sequence[int]], tgt_sequences: Sequence[Sequence[int]]):
        if len(src_sequences) != len(tgt_sequences):
            raise ValueError("Source and target sequences must have the same length.")
        self.src_sequences: List[List[int]] = [list(seq) for seq in src_sequences]
        self.tgt_sequences: List[List[int]] = [list(seq) for seq in tgt_sequences]

    def __len__(self) -> int:
        return len(self.src_sequences)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_sequences[idx], self.tgt_sequences[idx]
