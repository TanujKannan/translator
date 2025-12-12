import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        #FFX(x) = ReLU(xW1 + b1)W2 + b2

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

        self.gelu = nn.GELU()

    
    def forward(self, x):
        inter = self.gelu(self.w_1(x))

        inter = self.dropout(inter)

        output = self.w_2(inter)

        output = self.dropout(output)

        return output 
