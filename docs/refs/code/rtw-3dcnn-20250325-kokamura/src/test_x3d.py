from pathlib import Path

from lib import configure_logger
from x3d import dataloader
from x3d.config import Config
from x3d.model import X3DLightningModel, get_trainer

log = configure_logger(__name__)

if __name__ == "__main__":
    cfg = Config()

    dl_repo = dataloader.DataLoaderFactory(cfg)
    dl_repo.set_sample_times(sample_times=1)
    train_loader, _, test_loader = dl_repo.create_all_dataloaders()

    # n_features: 512
    # ckpt_path = Path("../exp/20241213_20h52m22s/ckpt/epoch=10-step=14520.ckpt")
    ckpt_path = Path("../exp/20241222_20h57m10s/ckpt/epoch=99-step=527800.ckpt")

    trainer = get_trainer(cfg, mode="test")
    with trainer.init_module(empty_init=True):
        x3d_model = X3DLightningModel.load_from_checkpoint(
            ckpt_path, cfg=cfg, n_train_batch=len(train_loader)
        )
    trainer.test(model=x3d_model, dataloaders=test_loader)
