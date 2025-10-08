from enum import Enum, auto
from typing import Optional

import lightning as L
import torch
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix

import utils
from cvt_custom.models.cmsm import ConstrainedMSM
from cvt_custom.models.msm import MultiMSM
from x3d.config import Config

from ..cnn3d.model import HeadOutput
from ..log import configure_logger
from ..typing import _LIGHTNING_MODEL, _SUBSPACE_MODEL

log = configure_logger(__name__)


class SubspaceMethodType(Enum):
    PRE_EXTRACTED = auto()
    REALTIME = auto()


class BaseSubspaceMethod(L.LightningModule):
    def __init__(
        self,
        cfg: Config,
        subspace_model: _SUBSPACE_MODEL,
        *,
        k_neighbors: int = 1,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.subspace_model = subspace_model
        self.k_neighbors = k_neighbors
        self.acc = Accuracy(task="multiclass", num_classes=cfg.n_classes)
        self.cm = MulticlassConfusionMatrix(num_classes=cfg.n_classes)

    def predict(self, features: torch.Tensor) -> int:
        features_np = features.squeeze(dim=0).cpu().numpy()

        assert isinstance(self.subspace_model, ConstrainedMSM | MultiMSM), ValueError(
            f"Supported subspace models are only ConstrainedMSM or MultiMSM, but got {type(self.subspace_model)}"
        )
        if isinstance(self.subspace_model, MultiMSM):
            pred = self.subspace_model.get_predictions(
                features_np, k_neighbors=self.k_neighbors
            )  # type: ignore
        elif isinstance(self.subspace_model, ConstrainedMSM):
            pred = self.subspace_model.get_predictions(features_np)  # type: ignore

        return pred.label

    def on_test_end(self) -> None:
        test_acc = self.acc.compute()
        log.info(f"Test Accuracy: {test_acc}")

        cm = self.cm.compute()
        utils.display_confusion_matrix(cm)
        fig_, ax_ = self.cm.plot()
        fig_.savefig(self.cfg.save_dir / "confusion_matrix.png")


class SubspaceMethodRealtime(BaseSubspaceMethod):
    def __init__(
        self,
        cfg: Config,
        subspace_model: _SUBSPACE_MODEL,
        k_neighbors: int,
        cnn3d_model: _LIGHTNING_MODEL,
    ) -> None:
        super().__init__(cfg, subspace_model)
        self.cnn3d_model = cnn3d_model
        self.cnn3d_model.eval()

    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        output = self.cnn3d_model.forward(inputs)
        assert isinstance(output, HeadOutput)
        return output

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        inputs, targets = batch
        with torch.no_grad():
            output = self.forward(inputs)
        assert output.features is not None

        pred = self.predict(output.features)
        pred_tensor = torch.tensor([pred])

        self.acc.update(pred_tensor, targets.view(-1).cpu())
        self.cm.update(
            pred_tensor.to(self.cfg.device), targets
        )  # GPU に移さないとエラーが出る


class SubspaceMethodFactory:
    @staticmethod
    def create(
        method_type: SubspaceMethodType,
        cfg: Config,
        subspace_model: _SUBSPACE_MODEL,
        *,
        k_neighbors: int,
        cnn3d_model: Optional[_LIGHTNING_MODEL] = None,
    ) -> BaseSubspaceMethod:
        if method_type == SubspaceMethodType.REALTIME:
            assert cnn3d_model is not None
            return SubspaceMethodRealtime(cfg, subspace_model, k_neighbors, cnn3d_model)

        raise ValueError(f"Unknown method type: {method_type}")
