import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(is_distribute: bool, local_rank: int):
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if is_distribute:
        return torch.device("cuda", local_rank)

    return torch.device("cuda")


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)  # type: ignore
