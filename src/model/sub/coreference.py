import torch
import torch.nn as nn

from base import BaseModel


class Coreference(BaseModel):
    """共参照スコアを計算"""
    def __init__(self,
                 bert_hidden_size: int,
                 dropout: float,
                 ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.out = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                sequence_output: torch.Tensor,  # (b, seq, hid)
                ) -> torch.Tensor:             # (b, seq, seq)
        batch_size, sequence_len, _ = sequence_output.size()

        h = self.linear(self.dropout(sequence_output))  # (b, seq, hid)
        h = torch.tanh(self.dropout(h.unsqueeze(1) + h.unsqueeze(2)))  # (b, seq, seq, hid)
        return self.out(h).squeeze(-1)  # (b, seq, seq)
