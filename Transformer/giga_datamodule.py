from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import GigaDataset
import TargetBuilder

class GigaDataModule(pl.LightningDataModule):

    def __init__(self,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 train_transform: Optional[Callable] = TargetBuilder(60, 60),
                 val_transform: Optional[Callable] = TargetBuilder(60, 60),
                 test_transform: Optional[Callable] = None,
                 **kwargs) -> None:
        super(GigaDataModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        ArgoverseV2Dataset(self.train_processed_dir, self.train_transform)
        ArgoverseV2Dataset(self.val_processed_dir, self.val_transform)
        ArgoverseV2Dataset(self.test_processed_dir, self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = ArgoverseV2Dataset(self.train_processed_dir,
                                                self.train_transform)
            self.val_dataset = ArgoverseV2Dataset(self.val_processed_dir,
                                              self.val_transform)
        if stage == "test" or stage is None:
            self.test_dataset = ArgoverseV2Dataset(self.test_processed_dir,
                                               self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
