"""Portable locations for private inputs, checkpoints, and generated results.

Defaults target an editable checkout. Set environment overrides before importing
this module when data or weights live elsewhere. Relative overrides resolve from
the checkout root, never from the current working directory. Importing this module
does not create directories or require access to private inputs.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_directory(variable: str, default: str) -> Path:
    """Resolve an environment override or a repository-relative default."""
    path = Path(os.environ.get(variable) or default).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


DATA_DIR = _configured_directory("AI_CPET_DATA_DIR", "data")
MODEL_WEIGHTS = _configured_directory("AI_CPET_MODELS_DIR", "models")
OUTPUT_DIR = _configured_directory("AI_CPET_OUTPUT_DIR", "outputs")

CLIN_CONTROL_DIR = DATA_DIR / "control_clinical.xlsx"
CLIN_OTS_DIR = DATA_DIR / "ots_clinical.xlsx"
CPET_CONTROL_DIR = DATA_DIR / "control_cpet.xlsx"
CPET_OTS_DIR = DATA_DIR / "ots_cpet.xlsx"
BBB_CONTROL_DIR = DATA_DIR / "control_bbb"
BBB_OTS_DIR = DATA_DIR / "ots_bbb"
ECG_CONTROL_DIR = DATA_DIR / "control_ecg"
ECG_OTS_DIR = DATA_DIR / "ots_ecg"


def require_file(path: str | Path) -> Path:
    """Return an existing input file or explain how to configure private inputs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {path}. See README.md for the input "
            "layout; set AI_CPET_DATA_DIR if the private study data lives elsewhere."
        )
    return path


def require_directory(path: str | Path) -> Path:
    """Return an existing input directory instead of silently loading no samples."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required input directory not found: {path}. See README.md for the "
            "input layout or set AI_CPET_DATA_DIR."
        )
    return path


def output_path(filename: str | Path, directory: str | Path | None = None) -> Path:
    """Create the output directory on demand and return a destination filename."""
    target = Path(directory) if directory is not None else OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target / filename
