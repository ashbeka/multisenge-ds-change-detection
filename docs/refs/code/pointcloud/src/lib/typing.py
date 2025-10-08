from typing import Literal

import lightning as L

from cvt_custom.models.base_class import SMBase

_MODE = Literal["train", "valid", "test"]
_SAMPLER_TYPE = Literal["random", "all", "fixed"]

_SUBSPACE_MODEL = SMBase
_SUBSPACE_MODEL_TYPE = type[_SUBSPACE_MODEL]

_LIGHTNING_MODEL = L.LightningModule
