from base import BaseDataLoader
from data_loader.dataset.dataset import PASDataset


class ConllDataLoader(BaseDataLoader):
    def __init__(self, dataset: PASDataset, batch_size: int, shuffle: bool, validation_split: float, num_workers: int):
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

    def add(self, dataset: PASDataset):
        self.dataset += dataset
        self.n_samples += len(dataset)
        # self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
