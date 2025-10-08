from typing import NamedTuple

import numpy as np


class PredResult(NamedTuple):
    label: int
    probs: np.ndarray
