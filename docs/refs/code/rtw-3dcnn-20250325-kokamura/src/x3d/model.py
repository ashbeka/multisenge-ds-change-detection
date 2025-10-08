import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.aggregation
import torchmetrics.classification
import torchmetrics.metric
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from pytorchvideo.models.x3d import ProjectedPool, create_x3d
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from lib import configure_logger
from lib.cnn3d.model import HeadOutput
from lib.typing import _MODE

from .config import Config

log = configure_logger(__name__)


def get_trainer(
    cfg: Config, *, mode: _MODE, logger: bool = False, precision: bool = True
) -> L.Trainer:
    if mode == "train":
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            save_dir=cfg.exp_dir,
            name=cfg.timestamp,
            offline=cfg.is_offline,
        )
        tb_logger = TensorBoardLogger(
            save_dir=cfg.exp_dir,
            name=cfg.timestamp,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            save_last=True,
            save_top_k=-1,
        )

        return L.Trainer(
            # fast_dev_run=50,  # todo: for debug
            max_epochs=cfg.epochs,
            logger=[wandb_logger, tb_logger],
            precision="bf16-mixed" if cfg.mixed_precision else None,
            strategy="ddp" if cfg.is_distributed else "auto",
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            callbacks=[checkpoint_callback],
        )

    elif mode == "valid" or mode == "test":
        _logger = False
        if logger:
            _logger = TensorBoardLogger(
                save_dir=cfg.exp_dir,
                name=cfg.timestamp,
            )
        return L.Trainer(
            logger=_logger,
            precision="bf16-mixed" if precision and cfg.mixed_precision else None,
        )


class X3DLightningModel(L.LightningModule):
    cfg: Config
    n_train_batch: int

    model: nn.Module
    train_loss: torchmetrics.Metric
    valid_acc: torchmetrics.Metric
    valid_loss: torchmetrics.Metric

    def __init__(self, cfg: Config, n_train_batch: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_train_batch = n_train_batch

        self.model = get_x3d_model(cfg, load_pretrained=True)

        self.train_loss = torchmetrics.aggregation.MeanMetric()
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=cfg.n_classes
        )
        self.valid_loss = torchmetrics.aggregation.MeanMetric()

        H, W, C = cfg.input_size
        self.example_input_array = torch.Tensor(cfg.batch_size, C, cfg.n_frames, H, W)

        torch.set_float32_matmul_precision("medium")

    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        return self.model(inputs)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, list[str]], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets, filenames = batch
        inputs = inputs.float()
        output = self.forward(inputs)
        loss = F.cross_entropy(output.preds, targets)

        self.train_loss.update(loss)

        self.log(
            "train/loss_step",
            self.train_loss.compute(),
            on_step=True,
            prog_bar=True,
            batch_size=inputs.size(0),
        )
        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],  # type: ignore
            on_step=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.train_loss.reset()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, list[str]], batch_idx: int
    ) -> None:
        inputs, targets, filenames = batch
        inputs = inputs.float()
        output = self.forward(inputs)

        # forward から削除したため、確率を取得する場合は softmax を適用する
        # probabilities = F.softmax(output.pred, dim=1)

        loss = F.cross_entropy(output.preds, targets)
        self.valid_loss.update(loss)
        self.valid_acc.update(output.preds.argmax(dim=1), targets)

    def on_validation_epoch_end(self) -> None:
        acc = self.valid_acc.compute()
        log.info(f"valid acc: {acc}")
        self.log("valid/acc", acc)
        self.valid_acc.reset()

        loss = self.valid_loss.compute()
        log.info(f"valid loss: {loss}")
        self.log("valid/loss", loss)
        self.valid_loss.reset()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_end(self) -> None:
        acc = self.valid_acc.compute()
        log.info(f"valid acc: {acc}")
        self.valid_acc.reset()

        loss = self.valid_loss.compute()
        log.info(f"valid loss: {loss}")
        self.valid_loss.reset()

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict]]:
        optimizer = SGD(
            self.parameters(),
            lr=self.cfg.lr,
            momentum=0.9,
            weight_decay=self.cfg.weight_decay,
        )

        total_steps = self.cfg.epochs * self.n_train_batch
        warmup_steps = self.cfg.epochs_warmup * self.n_train_batch
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=total_steps - warmup_steps,
            lr_min=self.cfg.lr_min,
            warmup_t=warmup_steps,
            warmup_lr_init=1e-4,  # type: ignore
            warmup_prefix=True,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)


class CustomX3DHead(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        dim_inner: int,
        dim_out: int,
        pool_kernel_size: tuple,
        n_features: int,
    ):
        super(CustomX3DHead, self).__init__()
        self.pool = ProjectedPool(
            pre_conv=nn.Conv3d(
                dim_in,
                dim_inner,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                bias=False,
            ),
            pre_norm=nn.BatchNorm3d(
                dim_inner,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            pre_act=nn.ReLU(),
            pool=nn.AvgPool3d(
                kernel_size=pool_kernel_size,
                stride=1,
                padding=0,
            ),
            post_conv=nn.Conv3d(
                dim_inner,
                n_features,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                bias=False,
            ),
            post_act=nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.proj = nn.Linear(in_features=n_features, out_features=dim_out, bias=True)
        # self.activation = nn.Softmax(dim=1)
        self.output_pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, x: torch.Tensor) -> HeadOutput:
        # Performs pooling.
        if self.pool is not None:
            x = self.pool(x)

        features = x.view(x.size(0), -1).detach()

        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)

        # Extract features before projection
        # [50, 512, 1, 1, 1]] => [50, 512]
        # features = x.squeeze(dim=(2, 3, 4)).detach()

        # Performs projection.
        if self.proj is not None:
            x = x.permute((0, 2, 3, 4, 1))
            x = self.proj(x)
            x = x.permute((0, 4, 1, 2, 3))
        # Performs activation.
        # if self.activation is not None:
        #     x = self.activation(x)

        if self.output_pool is not None:
            # Performs global averaging.
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)

        output = HeadOutput(preds=x, features=features)
        return output


def get_x3d_model(cfg: Config, load_pretrained: bool = True) -> nn.Module:
    model = create_x3d(
        input_channel=3,
        model_num_class=cfg.n_classes,
        input_clip_length=cfg.n_frames,
        input_crop_size=cfg.input_size[0],
    )

    if cfg.use_custom_head:
        if cfg.x3d_model == "x3d_s":
            kH = 4
            kW = 4
        elif cfg.x3d_model == "x3d_m":
            kH = 7
            kW = 7
        else:
            raise NotImplementedError

        model.blocks[-1] = CustomX3DHead(
            dim_in=cfg.dim_in,
            dim_inner=cfg.dim_inner,
            dim_out=cfg.n_classes,
            pool_kernel_size=(np.int64(cfg.n_frames), kH, kW),
            n_features=cfg.x3d_features,
        )

    if load_pretrained:
        x3d_pretrained_weights = torch.hub.load(
            "facebookresearch/pytorchvideo", cfg.x3d_model, pretrained=True
        )
        pretrained_dict = x3d_pretrained_weights.state_dict()  # type: ignore

        model_dict = model.state_dict()
        exclude_layers = ["blocks.5.proj", "blocks.5.pool.post_conv"]
        excluded_keys = []

        # 除外リストに含まれる層、または現在のモデルに存在しない層のキーを除外
        # 現在のモデルと事前学習済みモデルで、パラメータの形状が異なる場合
        for k, v in pretrained_dict.items():
            if (
                any(excluded_layer in k for excluded_layer in exclude_layers)
                or k not in model_dict
                or v.shape != model_dict[k].shape
            ):
                excluded_keys.append(k)
                continue
            model_dict[k] = v

        log.info(f"Excluded keys: {excluded_keys}")
        model.load_state_dict(model_dict, strict=False)

    return model
