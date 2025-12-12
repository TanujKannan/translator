import torch
import torch.nn as nn
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int,nhead: int,dropout: float = 0.1): 
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model) #Wq
        self.k_proj = nn.Linear(d_model, d_model) #Wk
        self.v_proj = nn.Linear(d_model, d_model) #Wv
        self.o_proj = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None):
        B, Lq, _ = query.shape
        q = self.q_proj(query)
        q = q.view(B, Lq, self.nhead, self.d_head).transpose(1,2)

        B, Lk, _ =  key.shape
        k = self.k_proj(key)
        k = k.view(B, Lk, self.nhead, self.d_head).transpose(1,2)

        B, Lv, _ = value.shape
        v = self.v_proj(value)
        v = v.view(B, Lv, self.nhead, self.d_head).transpose(1,2)

        scores = (q @ k.transpose(-2,-1))/math.sqrt(q.size(-1))

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn_out = weights @ v
        attn_out = attn_out.transpose(1,2).contiguous()
        attn_out = attn_out.view(B, Lq, self.d_model)
        out = self.o_proj(attn_out)
        return out
