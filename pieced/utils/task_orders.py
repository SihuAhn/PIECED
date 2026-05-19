"""Load class-to-task assignments from a JSON file."""
import json
import os
from typing import List

import torch


def load_task_splits(json_path: str, key: str) -> List[torch.Tensor]:
    """Return a list of per-task class-id tensors loaded from ``json_path[key]``.

    The JSON entries are dicts with keys like ``task_1``, ``task_2``, ...; they are
    returned in numeric order regardless of insertion order.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"task_orders JSON not found: {json_path}")

    with open(json_path) as f:
        all_splits = json.load(f)

    if key not in all_splits:
        raise KeyError(
            f"task_split_key '{key}' not found in {json_path}. "
            f"Available keys: {[k for k in all_splits if not k.startswith('_')]}"
        )

    split = all_splits[key]
    task_keys = sorted(
        (k for k in split if k.startswith("task_")),
        key=lambda x: int(x.split("_", 1)[1]),
    )
    return [torch.tensor(split[k], dtype=torch.long) for k in task_keys]
