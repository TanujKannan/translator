from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


class TranslationCollator:
    """
    Builds padded batches and masks for seq2seq/Transformer training.

    Assumes each example is (src_ids, tgt_ids) where tgt_ids already include
    start/end special tokens. Produces:
        - src:        [batch, src_len] long
        - tgt_input:  [batch, tgt_len] long (decoder inputs, without last token)
        - tgt_output: [batch, tgt_len] long (shifted targets, without first token)
        - src_pad_mask, tgt_pad_mask: [batch, seq_len] bool (True=token, False=pad)
        - tgt_causal_mask: [tgt_len, tgt_len] bool (True=masked-out future positions)
    """

    def __init__(
        self,
        pad_id: int,
        sos_id: int,
        eos_id: int,
        batch_first: bool = True,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.batch_first = batch_first
        self.device = device

    def __call__(self, batch: Sequence[Tuple[Sequence[int], Sequence[int]]]) -> Dict[str, torch.Tensor]:
        src_seqs, tgt_seqs = zip(*batch)

        # Targets must have at least <sos> and <eos> so we can shift.
        for tgt in tgt_seqs:
            if len(tgt) < 2:
                raise ValueError("Target sequence must include at least <sos> and <eos> for shifting.")

        src_tensors = [torch.tensor(seq, dtype=torch.long) for seq in src_seqs]
        tgt_tensors = [torch.tensor(seq, dtype=torch.long) for seq in tgt_seqs]

        src_padded = pad_sequence(src_tensors, batch_first=self.batch_first, padding_value=self.pad_id)

        # Shift targets: input drops the last token, output drops the first token.
        tgt_input_tensors = [t[:-1] for t in tgt_tensors]
        tgt_output_tensors = [t[1:] for t in tgt_tensors]

        tgt_input = pad_sequence(tgt_input_tensors, batch_first=self.batch_first, padding_value=self.pad_id)
        tgt_output = pad_sequence(tgt_output_tensors, batch_first=self.batch_first, padding_value=self.pad_id)

        # Padding masks: True where token is real (not pad).
        src_pad_mask = src_padded.ne(self.pad_id)
        tgt_pad_mask = tgt_input.ne(self.pad_id)

        # Causal mask for decoder self-attention: mask out future positions.
        tgt_len = tgt_input.size(1) if self.batch_first else tgt_input.size(0)
        tgt_causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.bool), diagonal=1)

        if self.device is not None:
            src_padded = src_padded.to(self.device)
            tgt_input = tgt_input.to(self.device)
            tgt_output = tgt_output.to(self.device)
            src_pad_mask = src_pad_mask.to(self.device)
            tgt_pad_mask = tgt_pad_mask.to(self.device)
            tgt_causal_mask = tgt_causal_mask.to(self.device)

        return {
            "src": src_padded,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output,
            "src_pad_mask": src_pad_mask,
            "tgt_pad_mask": tgt_pad_mask,
            "tgt_causal_mask": tgt_causal_mask,
        }
