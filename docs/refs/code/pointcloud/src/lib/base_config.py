import os
import socket
from datetime import datetime
from pathlib import Path

from .pytorch import get_device, set_seed


class BaseConfig:
    def __init__(self):
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

        self.project_name = "sthv2"
        self.timestamp = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
        self.prev_exp_timestamp = ""  # at the first run, this should be empty
        self.exp_dir = Path("../exp")

        self.num_workers = cpu_count if (cpu_count := os.cpu_count()) is not None else 1
        self.seed = 42
        set_seed(self.seed)

        self.world_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
        self.local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        self.world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))

        self.device = get_device(self.is_distributed, self.local_rank)
        self.is_supercomputer = socket.gethostname().startswith("bnode")

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_master_node(self) -> bool:
        return self.world_rank == 0
