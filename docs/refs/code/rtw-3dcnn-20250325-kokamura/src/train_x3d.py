import shutil

import utils
from lib import configure_logger
from x3d import dataloader
from x3d.config import Config
from x3d.model import X3DLightningModel, get_trainer

log = configure_logger(__name__)


def main() -> None:
    cfg = Config()
    cfg.sampler = "random"

    dl_repo = dataloader.DataLoaderFactory(cfg)
    train_loader, valid_loader, _ = dl_repo.create_all_dataloaders()
    x3d_model = X3DLightningModel(cfg=cfg, n_train_batch=len(train_loader))

    utils.makedirs(cfg.save_dir)
    if not (cfg.save_dir / "config.py").exists():
        shutil.copy("x3d/config.py", cfg.save_dir / "config.py")

    trainer = get_trainer(cfg, mode="train")
    trainer.fit(
        model=x3d_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=cfg.resume_ckpt_path if cfg.resume else None,
        # ref: https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state
    )


if __name__ == "__main__":
    main()
