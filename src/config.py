"""Paths, seed, thread count and cross-validation settings. The data folder is the local
`fds-pokemon-battles-prediction-2026/`, `FDS_DATA_DIR` or the Kaggle input directory."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_JOBS = int(os.environ.get("FDS_N_JOBS", "4"))
CV_SPLITS = 5
CV_REPEATS = 3


def _find_data_dir() -> Path:
    env = os.environ.get("FDS_DATA_DIR")
    if env:
        return Path(env)
    local = ROOT / "fds-pokemon-battles-prediction-2026"
    if (local / "train.jsonl").exists():
        return local
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted(kaggle.rglob("train.jsonl")):
            return p.parent
    raise FileNotFoundError("train.jsonl not found; set FDS_DATA_DIR")


def _dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


DATA_DIR = _find_data_dir()
CACHE_DIR = _dir(ROOT / "cache")
EXP_DIR = _dir(ROOT / "experiments")
SUB_DIR = _dir(ROOT / "submissions")
CONFIG_DIR = ROOT / "configs"
