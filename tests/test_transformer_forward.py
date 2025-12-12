import sys
from pathlib import Path

import torch

# Allow imports from the project root when running tests directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.collate import TranslationCollator
from model.transformer import Transformer


def test_transformer_forward_shape():
    torch.manual_seed(0)

    pad_id, sos_id, eos_id = 0, 1, 2
    vocab_size = 10

    # Two toy examples with <sos>/<eos> on targets.
    src_sequences = [
        [4, 5, 6],
        [7, 8],
    ]
    tgt_sequences = [
        [sos_id, 3, 4, eos_id],
        [sos_id, 5, eos_id],
    ]

    collator = TranslationCollator(
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
        batch_first=True,
    )
    batch = collator(list(zip(src_sequences, tgt_sequences)))

    model = Transformer(
        vocab_size=vocab_size,
        d_model=16,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=32,
        dropout=0.1,
        pad_id=pad_id,
        max_len=32,
    )

    logits = model(
        src=batch["src"],
        tgt_input=batch["tgt_input"],
        src_pad_mask=batch["src_pad_mask"],
        tgt_pad_mask=batch["tgt_pad_mask"],
        tgt_causal_mask=batch["tgt_causal_mask"],
    )

    assert logits.shape == (
        batch["tgt_input"].size(0),
        batch["tgt_input"].size(1),
        vocab_size,
    )
    assert logits.dtype == torch.float32
