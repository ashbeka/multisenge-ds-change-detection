from lib import configure_logger
from lib.subspace import SubspaceMethodType, fit_subspace, inference_subspace
from x3d.config import Config

log = configure_logger(__name__)


if __name__ == "__main__":
    cfg = Config()
    # cfg.n_classes = 20
    # cfg.x3d_features = 512

    subspace_model = fit_subspace(
        cfg,
        n_components=20,
        n_data_per_subspace=1,
    )
    inference_subspace(
        cfg,
        subspace_model,
        k_neighbors=5,
        method_type=SubspaceMethodType.PRE_EXTRACTED,
    )
