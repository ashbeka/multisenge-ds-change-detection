from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm

from cvt_custom.models import MultiMSM, MutualSubspaceMethod
from lib import configure_logger
from lib.features import load_cnn_features_1_class
from lib.typing import _SUBSPACE_MODEL, _SUBSPACE_MODEL_TYPE
from x3d.config import Config

from ._utils import load_subspace_model, save_subspace_model

log = configure_logger(__name__)


def fit_subspace(
    cfg: Config,
    n_components: int,
    n_data_per_subspace: int,
) -> _SUBSPACE_MODEL:
    log.info("n_components: %d", n_components)
    log.info("n_data_per_subspace: %d", n_data_per_subspace)

    subspace_model_cls: _SUBSPACE_MODEL_TYPE = MultiMSM
    name_pair: dict[str, str] = {
        MutualSubspaceMethod.__name__: "msm",
        MultiMSM.__name__: "multi_msm",
    }
    basename = name_pair[subspace_model_cls.__name__]

    filename_list = OrderedDict(
        {
            "basename": basename,
            "n_components": f"n{n_components}",
            "n_data_per_subspace": f"d{n_data_per_subspace}",
            "exp_tag": cfg.exp_tag,
        }
    )
    cache_dir = Path("_cache") / f"{basename}__n{n_components}_d{n_data_per_subspace}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cfg.is_supercomputer:
        n_use_files = -1
    else:
        n_use_files = 300
        filename_list["n_use_files"] = f"{n_use_files}_files"

    subspace_model_path = cfg.save_dir / ("_".join(filename_list.values()) + ".pkl")

    if subspace_model_path.exists():
        log.info("Already exists model.")
        log.info("Loading model from %s ...", subspace_model_path)
        subspace_model = load_subspace_model(subspace_model_path)
        return subspace_model

    log.info("Creating model %s ...", subspace_model_cls.__name__)

    cache_per_class = True
    subspace_model = subspace_model_cls(
        n_subdims=n_components,
        n_classes=cfg.n_classes,
        n_data_per_subspace=n_data_per_subspace,
        normalize=True,
        cache_per_class=cache_per_class,
        cache_dir=cache_dir,
    )

    if cache_per_class:
        if subspace_model.cache_dir.exists():
            for file in subspace_model.cache_dir.glob("*.npy"):
                file.unlink()
        else:
            subspace_model.cache_dir.mkdir(parents=True, exist_ok=True)

    n_files_map = {}
    for i in tqdm(range(cfg.n_classes), desc="Fitting model"):
        X_c, y_c = load_cnn_features_1_class(
            i,
            cfg.train_feature_dir,
            n_use_files=n_use_files,
            n_use_sample_times=-1,
        )
        n_files_map[y_c] = X_c.shape
        subspace_model.fit_1_class(X_c, y_c)

    if cache_per_class:
        subspace_model.fit_cache()

    for k, v in n_files_map.items():
        log.info(f"Class {k}: {v}")

    # Save model
    save_subspace_model(subspace_model, subspace_model_path)
    log.info("Model saved to %s", subspace_model_path)
    return subspace_model
