import os
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    Accuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassROC,
)
from tqdm import tqdm

from cvt_custom.models.cmsm import ConstrainedMSM
from cvt_custom.models.msm import MultiMSM
from cvt_custom.utils.typing import PredResult
from lib import configure_logger
from x3d import dataloader
from x3d.config import Config
from x3d.model import X3DLightningModel

from ..features import PreExtractedFeaturesDataset
from ..typing import _SUBSPACE_MODEL
from ._method import SubspaceMethodFactory, SubspaceMethodType

log = configure_logger(__name__)


class InferenceParallelProcessor:
    def __init__(
        self,
        cfg: Config,
        subspace_model: _SUBSPACE_MODEL,
        loader: DataLoader[PreExtractedFeaturesDataset],
        *,
        k_neighbors: Optional[int] = None,
    ):
        self.cfg = cfg
        self.subspace_model = subspace_model
        self.loader = loader
        self.k_neighbors = k_neighbors

        self.acc = Accuracy(task="multiclass", num_classes=cfg.n_classes)
        self.cm = MulticlassConfusionMatrix(num_classes=cfg.n_classes)
        self.roc = MulticlassROC(num_classes=cfg.n_classes)
        self.auroc = MulticlassAUROC(num_classes=cfg.n_classes)

        os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

    def predict(self, features: np.ndarray) -> PredResult:
        # NOTE: kwargs を渡す形にして呼び出しを共通化してもいいかもしれないけど、
        # 後々分からなくなりそうなので、ここでは別々に呼び出す

        if self.k_neighbors is None:
            # TODO: ConstrainedMSM のみに対応
            assert isinstance(self.subspace_model, ConstrainedMSM)
            result = self.subspace_model.get_predictions(features)
        else:
            # TODO: MultiMSM のみに対応
            assert isinstance(self.subspace_model, MultiMSM)
            result = self.subspace_model.get_predictions(
                features, k_neighbors=self.k_neighbors
            )
        return result

    def _process_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """joblib の並列処理用の関数

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): バッチデータ

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 予測結果と正解ラベル
        """
        features, target = batch
        features = features.squeeze(dim=0).cpu().numpy()
        pred = self.predict(features)
        pred_tensor = torch.tensor([pred.label])
        probs_tensor = torch.from_numpy(np.array([pred.probs]))
        return pred_tensor, target, probs_tensor

    def run_inference(self) -> None:
        """推論処理を並列実行する"""

        with torch.no_grad():
            """
            - backend=["loky", "multiprocessing"] の場合は、処理は早いが、
              なぜか torch.load の weights_only=True を求められるFutureWarningが出る
            - backend="threading" にするとFutureWarningは出てこないが、速度が遅い
            => loky でも処理自体には問題ないので、ignore に設定しておく
               warnings でも指定はできるが、メインスレッド以外が止まらず環境変数で指定
            """

            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(self._process_batch)(batch)
                for batch in tqdm(self.loader, desc="Processing batches")
            )

            preds = []
            targets = []
            probs_list = []
            for pred, target, probs in results:  # type: ignore
                preds.append(pred)
                targets.append(target)
                probs_list.append(probs)

            preds = torch.cat(preds).cpu()
            targets = torch.cat(targets).cpu()
            probs = torch.from_numpy(np.concatenate(probs_list, axis=0)).cpu()

            self.acc.update(preds, targets)
            self.cm.update(preds, targets)
            self.roc.update(probs, targets)
            self.auroc.update(probs, targets)

    def post_process(self) -> dict[str, Any]:
        """推論結果の後処理を行う

        Returns:
            dict: 推論結果の辞書 (accuracy, confusion_matrix)
        """
        test_acc = self.acc.compute()
        if isinstance(test_acc, torch.Tensor):
            test_acc = test_acc.item()
        log.info(f"Test Accuracy: {test_acc}")

        cm = self.cm.compute()
        print("Confusion Matrix:")
        print(cm.cpu().numpy())
        # utils.display_confusion_matrix(cm)

        # fig_, ax_ = self.cm.plot(cmap="jet")
        # fig_.savefig(self.cfg.save_dir / "confusion_matrix.png")
        return {
            "accuracy": test_acc,
            "confusion_matrix": cm,
            "roc": self.roc,  # インスタンスのまま返す
            "auroc": self.auroc.compute(),
        }


def inference_subspace(
    cfg: Config,
    subspace_model: _SUBSPACE_MODEL,
    k_neighbors: int,
    *,
    method_type: SubspaceMethodType = SubspaceMethodType.PRE_EXTRACTED,
) -> dict[str, Any]:
    log.info("k_neighbors: %d", k_neighbors)

    if method_type == SubspaceMethodType.PRE_EXTRACTED:
        test_loader = DataLoader(
            PreExtractedFeaturesDataset(cfg.test_feature_dir),
            batch_size=1,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        processor = InferenceParallelProcessor(
            cfg, subspace_model, test_loader, k_neighbors=k_neighbors
        )
        processor.run_inference()
        results = processor.post_process()
        return results

    elif method_type == SubspaceMethodType.REALTIME:
        # 現状並列処理はできない
        tester = L.Trainer(accelerator="gpu", precision=None)

        dl_repo = dataloader.DataLoaderFactory(cfg)
        train_loader, _, test_loader = dl_repo.create_all_dataloaders()
        with tester.init_module(empty_init=True):
            x3d_model = X3DLightningModel.load_from_checkpoint(
                cfg.save_dir / "ckpt" / "epoch=10-step=14520.ckpt",
                cfg=cfg,
                n_train_batch=len(train_loader),
            )

        subspace_method = SubspaceMethodFactory.create(
            method_type,
            cfg,
            subspace_model,
            k_neighbors=k_neighbors,
            cnn3d_model=x3d_model,
        )
        results = tester.test(model=subspace_method, dataloaders=test_loader)
        if isinstance(results, list):
            results = results[0]

        return dict(results)  # TODO: 未確認
