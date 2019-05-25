import torch
import torch.nn.functional as F


def cross_entropy_loss(output: torch.Tensor,  # (b, seq, case, seq)
                       target: torch.Tensor,  # (b, seq, case)
                       ) -> torch.Tensor:     # ()
    sequence_length = output.size(3)
    return F.cross_entropy(output.view(-1, sequence_length), target.view(-1), ignore_index=-1)
