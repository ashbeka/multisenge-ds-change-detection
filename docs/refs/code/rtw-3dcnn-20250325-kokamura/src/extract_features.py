import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib import configure_logger
from lib.features import FrameSamplingDataset, VideoLoaderFactory
from lib.typing import _LIGHTNING_MODEL, _MODE
from x3d import dataloader
from x3d.config import Config
from x3d.model import X3DLightningModel

log = configure_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["train", "test"], default="train"
    )
    return parser.parse_args()


@dataclass
class FeatureExtractor:
    cfg: Config
    cnn3d_model: _LIGHTNING_MODEL

    def run(
        self,
        loader: DataLoader[FrameSamplingDataset],
        *,
        is_save: bool = True,
    ):
        self.cnn3d_model.eval()
        dataset: FrameSamplingDataset = loader.dataset  # type: ignore

        features_buffer = []
        with tqdm(
            loader, total=len(dataset), leave=False, unit="samples", desc="Processing"
        ) as pbar:
            for inputs in pbar:
                inputs = inputs.float().to(self.cfg.device)

                with torch.no_grad():
                    output = self.cnn3d_model.forward(inputs)
                assert output.features is not None

                features_buffer.append(output.features.cpu())
                pbar.update(inputs.size(0))

            concat_features_np = torch.cat(features_buffer, dim=0).numpy()

            if is_save:
                fp = (
                    self.cfg.train_feature_dir
                    / str(dataset.target)
                    / f"{dataset.video_name}.npy"
                )
                fp.parent.mkdir(parents=True, exist_ok=True)
                np.save(fp, concat_features_np)


if __name__ == "__main__":
    cfg = Config()

    dl_repo = dataloader.DataLoaderFactory(cfg)
    dl_repo.set_sample_times(cfg.sample_times)

    log.info(f"Loading model from {cfg.ckpt_path_for_extract} ...")

    x3d_model = X3DLightningModel.load_from_checkpoint(
        cfg.ckpt_path_for_extract,
        cfg=cfg,
        n_train_batch=len(dl_repo.train_df // cfg.batch_size),
    )

    args = parse_args()
    mode_settings = {
        "train": {
            "feature_dir": cfg.train_feature_dir,
            "df": dl_repo.train_df,
            "params": dl_repo.train_params,
        },
        "test": {
            "feature_dir": cfg.test_feature_dir,
            "df": dl_repo.test_df,
            "params": dl_repo.test_params,
        },
    }

    if args.mode not in mode_settings:
        raise ValueError(f"Invalid mode: {args.mode}")

    mode: _MODE = args.mode  # type: ignore
    settings = mode_settings[mode]
    feature_dir = settings["feature_dir"]
    df = settings["df"]
    params = settings["params"]

    ds_factory = VideoLoaderFactory(df, mode, params)
    extractor = FeatureExtractor(cfg, cnn3d_model=x3d_model)

    with tqdm(
        ds_factory.generate_dataloaders(batch_size=10),
        total=len(df),
        desc="Extracting features",
    ) as pbar:
        for loader in pbar:
            video_name = loader.dataset.video_name  # type: ignore
            if any(feature_dir.glob(f"**/{video_name}.npy")):
                continue

            extractor.run(loader, is_save=True)
