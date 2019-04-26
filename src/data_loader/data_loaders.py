from base import BaseDataLoader
from data_loader.dataset import PASDataset


class ConlluDataLoader(BaseDataLoader):
    def __init__(self, dataset: PASDataset, batch_size: int, shuffle=True, validation_split=0.0, num_workers=1):
        super(ConlluDataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers)
