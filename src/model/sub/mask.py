import torch

from base import BaseModel


class Mask(BaseModel):
    """正解になり得ない項をマスクする"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                x: torch.Tensor,  # (b, seq, case, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,  # (b, seq, case, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = attention_mask.size()

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)

        return x + (~mask).float() * -1024.0  # (b, seq, case, seq)
