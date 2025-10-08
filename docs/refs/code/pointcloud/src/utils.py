from pathlib import Path
from typing import Literal

import numpy as np
import psutil
import torch
from rich.console import Console
from rich.table import Table


def makedirs(path: Path, exist_ok: bool = True):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=exist_ok)


def display_confusion_matrix(
    matrix: np.ndarray | torch.Tensor, *, title: str = "Confusion Matrix"
) -> None:
    table = Table(title=title, show_header=False, padding=(0, 0))

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()

    max_val = np.max(matrix)
    max_digits = len(str(int(max_val)))

    # Set the width of each column to the maximum number of digits
    for _ in range(matrix.shape[1]):
        table.add_column(justify="right", width=max_digits)

    for row in matrix:
        table.add_row(
            *[
                f"[on rgb({int(255 * (item / max_val))},0,0)]{item:{max_digits}}"
                if item > 0
                else f"{item:{max_digits}}"
                for item in row
            ]
        )

    console = Console()
    console.print(table)


def get_memory_usage_report(unit: Literal["GB", "MB"] = "GB") -> str:
    mapping = {
        "GB": 1024 * 1024 * 1024,
        "MB": 1024 * 1024,
    }
    memory_used = psutil.Process().memory_info().rss / mapping[unit]
    return f"Memory usage: {memory_used:.2f} {unit}"


def format_kv_pairs(kv: dict, delimiter: str = "_") -> str:
    return delimiter.join([f"{k}{v}" for k, v in kv.items()])
