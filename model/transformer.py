import torch
import torch.nn as nn

from .modules.positional import PositionalEmbedding
from .modules.encoder import Encoder
from .modules.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            d_model: int, 
            nhead: int,
            num_encoder_layers: int, 
            num_decoder_layers: int, 
            d_ff: int, 
            dropout: float, 
            pad_id: int, 
            max_len: int, 
            tie_embeddings: bool = False,
            ):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional = PositionalEmbedding(d_model, max_len=max_len, dropout=dropout)
        self.encoder = Encoder(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout, num_layers=num_encoder_layers)
        self.decoder = Decoder(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout, num_layers=num_decoder_layers)
        self.generator = nn.Linear(d_model, vocab_size)

        if tie_embeddings:
            self.generator.weight = self.tgt_embed.weight
    
    def forward(
       self,
       src: torch.Tensor,
       tgt_input: torch.Tensor,
       src_pad_mask: torch.Tensor,
       tgt_pad_mask: torch.Tensor,
       tgt_causal_mask: torch.Tensor,     
    ) -> torch.Tensor:
        src_emb = self.positional(self.src_embed(src))
        tgt_emb = self.positional(self.tgt_embed(tgt_input))

        memory = self.encoder(src_emb, src_key_padding_mask=~src_pad_mask)
        dec_out = self.decoder(
            tgt_emb,
            memory,
            tgt_key_padding_mask=~tgt_pad_mask,
            tgt_causal_mask=tgt_causal_mask,
            src_key_padding_mask=~src_pad_mask,
        )

        logits = self.generator(dec_out)
        return logits
