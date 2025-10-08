from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader

from lib import configure_logger
from lib.dataset import DatasetParams, MultiSampleDataset, load_tables
from lib.pytorch import worker_init_fn
from lib.typing import _MODE
from x3d.config import Config

log = configure_logger(__name__)


@dataclass
class DataLoaderFactory:
    cfg: Config

    train_df: pd.DataFrame = field(init=False)
    valid_df: pd.DataFrame = field(init=False)
    test_df: pd.DataFrame = field(init=False)

    train_params: DatasetParams = field(init=False)
    valid_params: DatasetParams = field(init=False)
    test_params: DatasetParams = field(init=False)

    def __post_init__(self) -> None:
        self.train_df, self.valid_df, self.test_df = load_tables(
            self.cfg.sthv2_dir, self.cfg.n_classes
        )
        log.info(
            "Dataset size: train: %s valid: %s test: %s",
            len(self.train_df),
            len(self.valid_df),
            len(self.test_df),
        )

        shared_params = DatasetParams(
            sample_times=1,
            sampler=self.cfg.sampler,  # type: ignore
            input_size=self.cfg.input_size,
            n_frames=self.cfg.n_frames,
            data_dir=self.cfg.sthv2_rawframes_dir,
        )
        self.train_params = shared_params.model_copy()
        self.valid_params = shared_params.model_copy()
        self.test_params = shared_params.model_copy()

    def create_all_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = self.create_dataloader(mode="train")
        valid_loader = self.create_dataloader(mode="valid")
        test_loader = self.create_dataloader(mode="test", batch_size=1)

        return train_loader, valid_loader, test_loader

    def create_dataloader(
        self,
        *,
        mode: _MODE = "train",
        drop_last: Optional[bool] = None,
        shuffle: Optional[bool] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ) -> DataLoader[MultiSampleDataset]:
        if mode == "train":
            df = self.train_df
            params = self.train_params
        elif mode == "valid":
            df = self.valid_df
            params = self.valid_params
        elif mode == "test":
            df = self.test_df
            params = self.test_params

        dataset = MultiSampleDataset(
            df,
            mode=mode,
            params=params,
        )

        if drop_last is None:
            drop_last = mode == "train"
        if shuffle is None:
            shuffle = mode == "train"
        if batch_size is None:
            batch_size = self.cfg.batch_size
        if num_workers is None:
            num_workers = self.cfg.num_workers
        if pin_memory is None:
            pin_memory = True

        return DataLoader(
            dataset,
            drop_last=drop_last,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def set_sample_times(self, sample_times: int) -> None:
        self.train_params.sample_times = sample_times
        self.valid_params.sample_times = sample_times
        self.test_params.sample_times = sample_times
