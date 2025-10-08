import re
from pathlib import Path

from rich.prompt import Prompt

from lib import configure_logger
from lib.base_config import BaseConfig

log = configure_logger(__name__)


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        is_force_offline = True
        self.is_offline = True if is_force_offline else self.is_supercomputer

        # * dataset
        self.sthv2_dir = Path("../data/sthv2")
        self.sthv2_rawframes_dir = self.sthv2_dir / "rawframes"
        self.n_classes = 174
        # self.n_classes = 20
        self.n_frames = 16

        self.input_size = (224, 224, 3)
        self.x3d_model = "x3d_m"
        # self.input_size = (128, 128, 3)
        # self.x3d_model = "x3d_s"

        self.sampler = "random"  # "random", "all", "fixed"

        # * dataloader
        self.sample_times = 1000

        # * model
        self.x3d_features = 2048
        # self.x3d_features = 512
        self.dim_in = 192
        self.dim_inner = 432
        self.use_custom_head = True  # extract features

        self.batch_size = 32
        self.accumulate_grad_batches = 1  # 8x2 = 16
        # self.batch_size = 8
        # self.accumulate_grad_batches = 2  # 8x2 = 16

        self.epochs = 100
        # self.epochs = 30
        self.epochs_warmup = 1
        self.patience = 5
        self.lr = 1e-3
        self.lr_min = 1e-6
        # self.weight_decay = 1e-2
        self.weight_decay = 5e-5  # ref: X3D
        self.clip_grad_norm = False
        self.max_grad_norm = 1e2
        self.accumulation_steps = 1  # 2
        self.mixed_precision = True

        # _target_exp_timestamp = "20241213_20h52m22s"  # nc: 20, x3d_feat: 512
        _target_exp_timestamp = "20241222_20h57m10s"  # nc: 174, x3d_feat: 2048

        if _target_exp_timestamp != "":
            self.prev_exp_timestamp = _target_exp_timestamp
            self.save_dir = self.exp_dir / _target_exp_timestamp
            if not self.save_dir.exists():
                raise FileNotFoundError(f"{self.save_dir} does not exist.")
        else:
            self.save_dir = self.exp_dir / self.timestamp

        self.checkpoint_dir = self.save_dir / "ckpt"

        # * resume
        self.resume = False
        self.resume_ckpt_path = self.checkpoint_dir / "last.ckpt"

        # * extract features
        self.extract_features = True
        self.ckpt_path_for_extract = self.checkpoint_dir / "last.ckpt"

        _exp_tag = "v3_174cls_prev_dropout"  # * optional tag  e.g. "_v1_20cls_..."
        # todo: かなり分かりづらいので要修正
        self._configure_features_dir(_exp_tag)
        self.resolve_features_dirname(is_interactive=False)

        # ** [update] n_classes, x3d_features
        # todo: 取得できるなら ckpt から取れると良さそう
        _update_n_classes = -1  # set to positive number to update, -1 to not update
        _update_x3d_features = -1  # set to positive number to update, -1 to not update

        if _update_n_classes > 0:
            self.n_classes = _update_n_classes
        if _update_x3d_features > 0:
            self.x3d_features = _update_x3d_features

        # val/test
        # self.output_dir = "../output"
        # self.metrics = "acc"
        # self.th = 0.3
        # self.test_time = 1

    def _configure_features_dir(self, exp_tag: str = "") -> None:
        self.exp_tag = exp_tag.strip("_")

        basename = f"features_{self.exp_tag}"
        self.train_feature_dir = self.save_dir / basename
        self.test_feature_dir = self.save_dir / f"test_{basename}"

    def _resolve_exp_tag(self, features_dirname: str) -> str:
        res = re.search(r"features_(.+)", features_dirname)
        exp_tag = res.group(1) if res else ""
        return exp_tag

    def resolve_features_dirname(self, is_interactive: bool = False) -> str:
        if self.train_feature_dir.exists():
            return self.train_feature_dir.name

        log.warning(f"{self.train_feature_dir} does not exist.")

        dirs = [dir_.name for dir_ in self.save_dir.glob("features*")]
        if len(dirs) == 0:
            log.error("No features dir found.")
            raise FileNotFoundError("No features dir found in %s", self.save_dir)

        features_dirname = dirs[0]
        if len(dirs) > 1:
            if not is_interactive:
                log.error("Multiple features dir found.")
                raise FileNotFoundError(
                    "Multiple features dir found in %s", self.save_dir
                )

            features_dirname = Prompt.ask(
                "Select features dir", choices=dirs, default=dirs[0]
            )

        exp_tag = self._resolve_exp_tag(features_dirname)
        self._configure_features_dir(exp_tag)

        log.info("update cfg.train_feature_dir: %s", self.train_feature_dir)
        log.info("update cfg.test_feature_dir: %s", self.test_feature_dir)

        return features_dirname
