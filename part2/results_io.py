"""Uniform on-disk layout for every tool's outputs under part2/results/."""
import json
from pathlib import Path

import pandas as pd

from part2 import config


def _path(subdir: str, filename: str) -> Path:
    folder = config.RESULTS_DIR / subdir
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def printable_params(params: dict) -> dict:
    return {k: ("none" if v is None else v) for k, v in params.items()}


def save_predictions(tool, experiment, y_true, y_prob) -> Path:
    path = _path("predictions", f"{tool}_{experiment}.csv")
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(path, index=False)
    return path


def save_run(tool, experiment, info: dict) -> Path:
    path = _path("runs", f"{tool}_{experiment}.json")
    path.write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
    return path


def save_table(df: pd.DataFrame, subdir: str, filename: str) -> Path:
    path = _path(subdir, filename)
    df.to_csv(path, index=False)
    return path


def save_text(text: str, subdir: str, filename: str) -> Path:
    path = _path(subdir, filename)
    path.write_text(text, encoding="utf-8")
    return path
