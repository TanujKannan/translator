import torch.nn as nn
from .feedforward import FFN
from .attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

        self.ffn = FFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, src_key_padding_mask=None):

        x_norm = self.norm1(x)

        attn_out = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask)

        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)

        ff_output = self.ffn(x_norm)

        x = x + self.dropout2(ff_output)

        return x

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout = 0.1, num_layers = 6):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, src_key_padding_mask = None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

