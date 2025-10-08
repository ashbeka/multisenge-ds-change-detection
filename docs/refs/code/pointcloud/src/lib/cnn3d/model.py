from typing import NamedTuple, Optional

import torch


class HeadOutput(NamedTuple):
    preds: torch.Tensor
    features: Optional[torch.Tensor] = None
