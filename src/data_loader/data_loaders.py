from typing import List, Tuple

import torch
import numpy as np
from base import BaseDataLoader
from data_loader.dataset.pas_dataset import PASDataset
from torch.utils.data import Sampler, BatchSampler


class PASDataLoader(BaseDataLoader):
    def __init__(self,
                 dataset: PASDataset,
                 batch_size: int,
                 shuffle: bool,
                 validation_split: float,
                 num_workers: int,
                 ):
        super().__init__(dataset,
                         batch_size,
                         shuffle,
                         validation_split,
                         num_workers,
                         collate_fn=broadcast_collate_fn,
                         sampler=None)


def broadcast_collate_fn(batch: List[Tuple[np.ndarray, ...]]) -> Tuple[torch.Tensor, ...]:
    input_ids, input_mask, segment_ids, target, ng_token_mask, deps, task, overt_mask = zip(*batch)
    target = np.broadcast_arrays(*target)
    ng_token_mask = np.broadcast_arrays(*ng_token_mask)
    deps = np.broadcast_arrays(*deps)
    transposed = (input_ids, input_mask, segment_ids, target, ng_token_mask, deps, task, overt_mask)
    return tuple(torch.as_tensor(np.stack(elem, axis=0)) for elem in transposed)
