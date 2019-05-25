from base import BaseDataLoader
from data_loader.dataset import PASDataset


class ConlluDataLoader(BaseDataLoader):
    def __init__(self, dataset: PASDataset, batch_size: int, shuffle: bool, validation_split: float, num_workers: int):
        super(ConlluDataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers)
