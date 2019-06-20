import torch
import torch.nn.functional as F


def cross_entropy_pas_loss(output: torch.Tensor,  # (b, seq, case, seq)
                           target: torch.Tensor,  # (b, seq, case)
                           *_
                           ) -> torch.Tensor:     # ()
    sequence_length = output.size(3)
    return F.cross_entropy(output.view(-1, sequence_length), target.view(-1), ignore_index=-1)


def cross_entropy_pas_dep_loss(output: torch.Tensor,  # (b, seq, case+1, seq)
                               target: torch.Tensor,  # (b, seq, case)
                               dep: torch.Tensor,     # (b, seq, seq)
                               ) -> torch.Tensor:     # ()
    sequence_length = output.size(3)
    output_pas = output[:, :, :-1, :].contiguous()  # (b, seq, case, seq)
    pas_loss = F.cross_entropy(output_pas.view(-1, sequence_length), target.view(-1), ignore_index=-1)
    dep_softmax = torch.softmax(output[:, :, -1, :], dim=2)  # (b, seq, seq)
    dep_loss = torch.sum(-torch.log(dep_softmax) * dep.float())
    return pas_loss + dep_loss
