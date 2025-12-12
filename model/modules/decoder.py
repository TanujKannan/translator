import torch.nn as nn
from .feedforward import FFN
from .attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)

        self.ffn = FFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x,
        enc_output,
        tgt_key_padding_mask=None,
        tgt_causal_mask=None,
        src_key_padding_mask=None,
    ):

        x_norm = self.norm1(x)

        attn_1 = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=tgt_causal_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + self.dropout1(attn_1)

        x_norm = self.norm2(x)
        attn_2 = self.cross_attn(
            x_norm,
            enc_output,
            enc_output,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.dropout2(attn_2)

        x_norm = self.norm3(x)
        ff_output = self.ffn(x_norm)
        x = x + self.dropout3(ff_output)

        return x

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout = 0.1, num_layers = 6):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])

    def forward(
        self,
        x,
        enc_output,
        tgt_key_padding_mask=None,
        tgt_causal_mask=None,
        src_key_padding_mask=None,
    ):
        for layer in self.layers:
            x = layer(
                x,
                enc_output,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_causal_mask=tgt_causal_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return x
