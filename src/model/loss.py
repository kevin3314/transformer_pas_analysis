from typing import Tuple

import torch
import torch.nn.functional as F


def cross_entropy_pas_loss(output: torch.Tensor,  # (b, seq, case, seq)
                           target: torch.Tensor,  # (b, seq, case, seq)
                           *_
                           ) -> torch.Tensor:     # ()
    log_softmax = torch.log_softmax(output, dim=3)  # (b, seq, case, seq)
    eps = 1e-6
    return torch.sum(-log_softmax * target) / (torch.sum(target) + eps)


def multi_cross_entropy_pas_loss(output: Tuple[torch.Tensor],  # (b, seq, case, seq)
                                 target: torch.Tensor,  # (b, seq, case, seq)
                                 *_
                                 ) -> torch.Tensor:     # ()
    total_loss = None
    for out in output:
        loss = cross_entropy_pas_loss(out, target)
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss
    return total_loss


def cross_entropy_pas_dep_loss(output: torch.Tensor,  # (b, seq, case+1, seq)
                               target: torch.Tensor,  # (b, seq, case, seq)
                               dep: torch.Tensor,     # (b, seq, seq)
                               ) -> torch.Tensor:     # ()
    output_pas = output[:, :, :-1, :].contiguous()  # (b, seq, case, seq)
    pas_loss = cross_entropy_pas_loss(output_pas, target)
    dep_softmax = torch.softmax(output[:, :, -1, :], dim=2)  # (b, seq, seq)
    dep_loss = torch.sum(-torch.log(dep_softmax) * dep.float())
    return pas_loss + dep_loss
