from typing import Tuple

import torch

from utils.constants import TASK_ID


def cross_entropy_pas_loss(output: torch.Tensor,  # (b, seq, case, seq)
                           target: torch.Tensor,  # (b, seq, case, seq)
                           *_
                           ) -> torch.Tensor:     # ()
    log_softmax = torch.log_softmax(output, dim=3)  # (b, seq, case, seq)
    eps = 1e-6
    return torch.sum(-log_softmax * target) / (torch.sum(target) + eps)


def multi_cross_entropy_pas_loss(output: Tuple[torch.Tensor, torch.Tensor],  # (b, seq, case, seq)
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


def cross_entropy_pas_dep_loss(output: Tuple[torch.Tensor, torch.Tensor],  # (b, seq, case, seq), (b, seq, seq)
                               target: torch.Tensor,  # (b, seq, case, seq)
                               dep: torch.Tensor,     # (b, seq, seq)
                               *_
                               ) -> torch.Tensor:     # ()
    output_pas, output_dep = output
    pas_loss = cross_entropy_pas_loss(output_pas, target)  # ()
    log_softmax = torch.log_softmax(output_dep, dim=2)  # (b, seq, seq)
    dep_loss = torch.sum(-log_softmax * dep)  # ()
    return pas_loss + dep_loss


def cross_entropy_pas_commonsense_loss(output: Tuple[torch.Tensor, torch.Tensor],  # (b, seq, case, seq), (b, 2)
                                       target: torch.Tensor,  # (b, seq, case, seq) or (b, 1, 1, 1)
                                       _,
                                       task: torch.Tensor,    # (b)
                                       ) -> torch.Tensor:     # ()
    mask_pa = task.eq(TASK_ID['pa'])
    mask_ci = task.eq(TASK_ID['ci'])
    if mask_pa.sum().item() > 0:
        output_pas = output[0][mask_pa, :, :, :]  # (x, seq, case, seq)
        target_pas = target[task == TASK_ID['pa'], :, :, :]  # (x, seq, case, seq) or (0, 1, 1, 1)
        loss_pas = cross_entropy_pas_loss(output_pas, target_pas)  # ()
    else:
        loss_pas = torch.tensor(0, device=target.device)

    if mask_ci.sum().item() > 0:
        output_commonsense = output[1][task == TASK_ID['ci'], :]  # (b-x, 2)
        log_softmax_commonsense = torch.log_softmax(output_commonsense, dim=1)  # (b-x, 2)
        target_commonsense = target[task == TASK_ID['ci'], 0, 0, 0]  # (b-x)
        loss_commonsense = -log_softmax_commonsense[torch.arange(target_commonsense.size(0)), target_commonsense].mean()
    else:
        loss_commonsense = torch.tensor(0, device=target.device)

    return loss_pas + loss_commonsense
