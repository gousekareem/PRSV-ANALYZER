from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that converts NumPy values into Python-native values.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """
    Save data to JSON using a NumPy-safe encoder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, cls=NumpyJSONEncoder, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """
    Load JSON from disk.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)