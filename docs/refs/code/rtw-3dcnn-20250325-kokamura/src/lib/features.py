import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import DatasetParams, get_transform, sample_frames
from .log import configure_logger
from .pytorch import worker_init_fn
from .typing import _MODE

log = configure_logger(__name__)


class FrameSamplingDataset(Dataset):
    sampled_frame_indices: list[list[int]]
    video_name: str
    target: int
    mode: _MODE
    params: DatasetParams
    aug: dict[str, A.BasicTransform]

    def __init__(
        self,
        sampled_frame_indices: list[list[int]],
        video_name: str,
        target: int,
        *,
        mode: _MODE,
        params: DatasetParams,
    ):
        self.sampled_frame_indices = sampled_frame_indices
        self.video_name = video_name
        self.target = target
        self.mode = mode
        self.params = params

        self.aug = get_transform(params.input_size)

    def __len__(self):
        return len(self.sampled_frame_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        indices = self.sampled_frame_indices[idx]
        sampled_imgs = self._process_frames(self.video_name, indices)
        return sampled_imgs

    def _process_frames(self, video_stem: str, indices: list[int]) -> torch.Tensor:
        optional_aug = np.random.uniform() >= 0.5

        imgs = []  # (n_frames, H, W, C)
        for frame_idx in indices:
            filepath = self.params.data_dir / video_stem / f"img_{frame_idx:05d}.jpg"
            img = cv2.imread(str(filepath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.aug["resize"](image=img)["image"]
            if self.mode == "train":
                img = self.aug["color"](image=img)["image"]
                if optional_aug:
                    img = self.aug["flip"](image=img)["image"]

            imgs.append(torch.tensor(img))

        # permute (n_frames, H, W, C) -> (C, n_frames, H, W)
        sampled_imgs = np.array(imgs)
        sampled_imgs = torch.from_numpy(sampled_imgs).permute(3, 0, 1, 2)

        return sampled_imgs


@dataclass
class VideoLoaderFactory:
    df: pd.DataFrame
    mode: _MODE
    params: DatasetParams

    def _generate_batches(self) -> Generator[FrameSamplingDataset, None, None]:
        """1データごとに FrameSamplingDataset を生成

        Yields:
            Generator[FrameSamplingDataset, None, None]: 1データごとの FrameSamplingDataset
        """
        for _, row in self.df.iterrows():
            video_name = str(row["video"])
            target = int(row["target"])
            max_frame = int(row["max_frame"])

            sampled_frame_indices = sample_frames(
                n_frames=self.params.n_frames,
                max_frame=max_frame,
                sample_times=self.params.sample_times,
                sampler=self.params.sampler,
            )

            yield FrameSamplingDataset(
                sampled_frame_indices,
                video_name,
                target,
                mode=self.mode,
                params=self.params,
            )

    def generate_dataloaders(
        self,
        *,
        drop_last: bool = False,
        shuffle: bool = False,
        batch_size: int = 100,
        num_workers: int = n if (n := os.cpu_count()) else 1,
        pin_memory: bool = True,
    ) -> Generator[DataLoader, None, None]:
        """ミニバッチを並列処理する DataLoader を生成

        Args:
            drop_last (bool, optional): Defaults to False.
            shuffle (bool, optional): Defaults to False.
            batch_size (int, optional): Defaults to 100.
            num_workers (int, optional): Defaults to os.cpu_count().
            pin_memory (bool, optional): Defaults to True.

        Yields:
            Generator[DataLoader, None, None]: 1データごとの DataLoader
        """
        for dataset in self._generate_batches():
            yield DataLoader(
                dataset,
                drop_last=drop_last,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn,
            )


def load_cnn_features(
    n_classes: int,
    load_dir: Path,
    *,
    n_use_files: int = -1,
    n_use_sample_times: int = -1,
) -> tuple[list[np.ndarray], np.ndarray]:
    features = []
    labels = []

    for class_label in range(n_classes):
        class_features, class_label = load_cnn_features_1_class(
            class_label,
            load_dir,
            n_use_files=n_use_files,
            n_use_sample_times=n_use_sample_times,
        )
        features.append(class_features)
        labels.append(class_label)

    return features, np.unique(labels)


def load_cnn_features_1_class(
    class_label: int,
    load_dir: Path,
    *,
    n_use_files: int = -1,
    n_use_sample_times: int = -1,
) -> tuple[np.ndarray, int]:
    class_dir = load_dir / str(class_label)
    class_features = []

    for i, file in enumerate(class_dir.glob("*.npy")):
        if 0 <= n_use_files <= i:
            # log.warning(f"Class {class_label}: use only {n_use_files} files")
            break
        features = np.load(file, allow_pickle=False)
        if 0 <= n_use_sample_times:
            features = features[:n_use_sample_times]
        class_features.append(features)

    # if n_use_files != -1 and n_use_files != len(class_features):
    #     log.warning(f"Class {class_label}: use only {n_use_files} files")

    return np.array(class_features), class_label


def load_cnn_features_1_class_memmap(
    class_label: int,
    load_dir: Path,
    *,
    n_use_files: int = -1,
    n_use_sample_times: int = -1,
) -> tuple[list[np.memmap], int]:  # np.ndarray → list[np.memmap] に変更
    class_dir = load_dir / str(class_label)
    class_features = []

    for i, file in enumerate(class_dir.glob("*.npy")):
        if 0 <= n_use_files <= i:
            break
        memmap_array = np.memmap(file, mode="r")

        # 必要に応じてサンプル数を制限
        if 0 <= n_use_sample_times:
            memmap_array = memmap_array[:n_use_sample_times]

        class_features.append(memmap_array)
    return class_features, class_label


class PreExtractedFeaturesDataset(Dataset):
    def __init__(self, features_dir: Path):
        self.features_dir = features_dir

        self.npy_features_paths: list[str] = []
        self.labels: list[int] = []

        # directory name is class label  (e.g. 0, 1, 2, ...)
        self.classes = sorted(
            [int(d.name) for d in self.features_dir.iterdir() if d.is_dir()]
        )
        for class_label in self.classes:
            class_dir = self.features_dir / str(class_label)

            for file in class_dir.glob("*.npy"):
                self.npy_features_paths.append(str(file))
                self.labels.append(class_label)

    def __len__(self):
        return len(self.npy_features_paths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        features_path = self.npy_features_paths[idx]
        label = self.labels[idx]

        features = np.load(features_path, allow_pickle=False)
        return features, label
