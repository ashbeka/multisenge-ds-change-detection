import random
from pathlib import Path
from typing import Generator

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from pydantic import BaseModel
from torch.utils.data import Dataset

from .log import configure_logger
from .typing import _MODE, _SAMPLER_TYPE

log = configure_logger(__name__)


def load_tables(data_dir: Path, use_n_classes: int = -1):
    # set input paths
    input_dir = data_dir / "annotations"
    train_processed_csv = input_dir / "something-something-v2-train-processed.csv"
    valid_processed_csv = input_dir / "something-something-v2-valid-processed.csv"
    test_processed_csv = input_dir / "something-something-v2-test-processed.csv"

    train_df = pd.read_csv(train_processed_csv)
    valid_df = pd.read_csv(valid_processed_csv)
    test_df = pd.read_csv(test_processed_csv)

    if use_n_classes > 0:
        train_df = train_df.loc[(train_df["target"] < use_n_classes)]
        valid_df = valid_df.loc[(valid_df["target"] < use_n_classes)]
        test_df = test_df.loc[(test_df["target"] < use_n_classes)]

    return train_df, valid_df, test_df


def get_transform(input_size: tuple[int, int, int]):
    train_aug = A.Compose(
        [A.ShiftScaleRotate(p=0.5), A.Normalize(mean=[0.0], std=[1.0]), ToTensorV2()]
    )
    test_aug = A.Compose([A.Normalize(mean=[0.0], std=[1.0]), ToTensorV2()])
    resize_aug = A.Resize(input_size[0], input_size[1])
    flip_aug = A.HorizontalFlip(p=1)
    color_aug = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1),
                    A.HueSaturationValue(p=1),
                ],
                p=0.5,
            ),
        ]
    )

    aug = {
        "train": train_aug,
        "test": test_aug,
        "resize": resize_aug,
        "color": color_aug,
        "flip": flip_aug,
    }

    return aug


def sample_frames(
    n_frames: int,
    max_frame: int,
    sample_times: int = 1,
    sampler: _SAMPLER_TYPE = "random",
) -> list[list[int]]:
    multi_sampled_frames = []

    if sampler == "all":
        for i in range(sample_times):
            frames = list(range(1, (max_frame + 1)))
            multi_sampled_frames.append(frames)
    elif sampler == "fixed":
        for i in range(sample_times):
            frames = list(range(1, (max_frame + 1)))
            if max_frame < n_frames:
                repeats = n_frames // max_frame
                remainder = n_frames % max_frame
                frames = frames * repeats + frames[:remainder]
            elif max_frame > n_frames:
                frames = frames[:n_frames]
            multi_sampled_frames.append(frames)
    else:  # elif sampler=='random':
        for i in range(sample_times):
            if max_frame > n_frames - 1:
                frames = sorted(random.sample(range(1, max_frame + 1), n_frames))
            else:
                # log.debug("max_frame < n_frames")
                frames = sorted(
                    list(range(1, max_frame + 1))
                    + random.choices(range(1, max_frame + 1), k=n_frames - max_frame)
                )
            multi_sampled_frames.append(frames)

    return multi_sampled_frames


class DatasetParams(BaseModel):
    sampler: _SAMPLER_TYPE
    sample_times: int
    input_size: tuple[int, int, int]
    n_frames: int
    data_dir: Path


class MultiSampleDataset(Dataset):
    df: pd.DataFrame
    mode: _MODE
    params: DatasetParams

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        mode: _MODE = "test",
        params: DatasetParams,
    ):
        self.df = df
        self.videos = df.video.values
        self.max_frames = df.max_frame.values
        self.targets = df.target.values

        self.mode = mode
        self.params = params

        self.aug = get_transform(params.input_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Returns a tuple of (sampled_imgs, target, video_stem)
        sampled_imgs:
            (sample_times, C, n_frames, H, W) if sample_times > 1
            (C, n_frames, H, W) if sample_times == 1
        target:
            (sample_times) if sample_times > 1
            scalar if sample_times == 1
        video_stem:
            str

        Args:
            idx (int): index of the dataset

        Returns:
            tuple[torch.Tensor, torch.Tensor, str]: batch of images, target, video_stem
        """
        video_stem = str(self.videos[idx])
        max_frame = int(self.max_frames[idx])
        target = torch.tensor(int(self.targets[idx]))

        multi_sampled_frames = sample_frames(
            n_frames=self.params.n_frames,
            max_frame=max_frame,
            sample_times=self.params.sample_times,
            sampler=self.params.sampler,
        )

        sampled_imgs = self._process_frames(video_stem, multi_sampled_frames)

        if self.params.sample_times == 1:
            # squeeze (sample_times, C, n_frames, H, W) -> (C, n_frames, H, W)
            sampled_imgs = sampled_imgs.squeeze(dim=0)
        else:
            # repeat target (sample_times)
            target = target.repeat(self.params.sample_times)
        return sampled_imgs, target, video_stem

    def generate_batches(
        self, idx: int, *, minibatch_size: int = 100
    ) -> Generator[tuple[torch.Tensor, torch.Tensor, str], None, None]:
        """Generates batches of samples, splitting sample_times into smaller chunks.

        Args:
            idx (int): Index of the dataset.
            batch_size (int): The desired batch size (number of sample_times to process in each batch).

        Yields:
            tuple[torch.Tensor, torch.Tensor, str]: A batch of (sampled_imgs, target, video_stem).
        """
        video_stem = str(self.videos[idx])
        max_frame = int(self.max_frames[idx])
        target = torch.tensor(int(self.targets[idx]))

        # メモリからフレームを取得
        frames_data = self.load_and_cache_frames(idx)

        multi_sampled_frames = sample_frames(
            n_frames=self.params.n_frames,
            max_frame=max_frame,
            sample_times=self.params.sample_times,
            sampler=self.params.sampler,
        )

        for i in range(0, self.params.sample_times, minibatch_size):
            sampled_frames_batch = multi_sampled_frames[i : i + minibatch_size]
            sampled_imgs = self._process_cached_frames(
                frames_data, sampled_frames_batch
            )

            if len(sampled_frames_batch) == 1:
                sampled_imgs = sampled_imgs.squeeze(dim=0)

            target_batch = target.repeat(len(sampled_frames_batch))
            yield sampled_imgs, target_batch, video_stem

    def _process_cached_frames(
        self, frames_data: list[np.ndarray], sampled_frames_batch: list[list[int]]
    ) -> torch.Tensor:
        """__getitem__ と generate_batches で共通の画像処理部分"""
        optional_aug = np.random.uniform() >= 0.5
        multi_sampled_imgs = []
        for frames in sampled_frames_batch:
            imgs = []  # (n_frames, H, W, C)
            for frame in frames:
                img = frames_data[frame - 1]
                # filepath = self.params.data_dir / video_stem / f"img_{frame:05d}.jpg"
                # img = cv2.imread(str(filepath))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = self.aug["resize"](image=img)["image"]
                if self.mode == "train":
                    img = self.aug["color"](image=img)["image"]
                    if optional_aug:
                        img = self.aug["flip"](image=img)["image"]

                imgs.append(torch.tensor(img))
            multi_sampled_imgs.append(imgs)

        # permute (sample_times, n_frames, H, W, C) -> (sample_times, C, n_frames, H, W)
        multi_sampled_imgs = np.array(multi_sampled_imgs)
        sampled_imgs = torch.from_numpy(multi_sampled_imgs).permute(0, 4, 1, 2, 3)

        return sampled_imgs

    def _process_frames(
        self, video_stem: str, sampled_frames_batch: list[list[int]]
    ) -> torch.Tensor:
        """__getitem__ と generate_batches で共通の画像処理部分"""
        optional_aug = np.random.uniform() >= 0.5
        multi_sampled_imgs = []
        for frames in sampled_frames_batch:
            imgs = []  # (n_frames, H, W, C)
            for frame in frames:
                filepath = self.params.data_dir / video_stem / f"img_{frame:05d}.jpg"
                img = cv2.imread(str(filepath))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = self.aug["resize"](image=img)["image"]
                if self.mode == "train":
                    img = self.aug["color"](image=img)["image"]
                    if optional_aug:
                        img = self.aug["flip"](image=img)["image"]

                imgs.append(torch.tensor(img))
            multi_sampled_imgs.append(imgs)

        # permute (sample_times, n_frames, H, W, C) -> (sample_times, C, n_frames, H, W)
        multi_sampled_imgs = np.array(multi_sampled_imgs)
        sampled_imgs = torch.from_numpy(multi_sampled_imgs).permute(0, 4, 1, 2, 3)

        return sampled_imgs

    def load_and_cache_frames(self, idx: int) -> list[np.ndarray]:
        """指定されたインデックスの動画のフレームを全て読み込み、キャッシュする"""
        video_stem = str(self.videos[idx])
        max_frame = int(self.max_frames[idx])

        video_frames = []
        for frame in range(1, max_frame + 1):
            filepath = self.params.data_dir / video_stem / f"img_{frame:05d}.jpg"
            img = cv2.imread(str(filepath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_frames.append(img)

        return video_frames
