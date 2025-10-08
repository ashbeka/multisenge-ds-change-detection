import copy
import pickle
from pathlib import Path

from ..log import configure_logger
from ..typing import _SUBSPACE_MODEL

log = configure_logger(__name__)


def save_subspace_model(model: _SUBSPACE_MODEL, save_path: Path) -> None:
    """SubspaceMethodモデルを保存

    Args:
        model: 保存するモデル
        save_path: 保存先のパス
    """
    model_copy = copy.deepcopy(model)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model_copy, f)

    log.info(f"Model saved to {save_path}")


def load_subspace_model(load_path: Path) -> _SUBSPACE_MODEL:
    """SubspaceMethodモデルを読み込み

    Args:
        load_path: 読み込むモデルのパス

    Returns:
        読み込んだモデル。失敗時はNone
    """
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")

    with open(load_path, "rb") as f:
        model = pickle.load(f)

    log.info(f"Model loaded from {load_path}")
    return model
