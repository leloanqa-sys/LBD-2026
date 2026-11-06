"""Load nested configuration without lossy parser fallbacks."""
from pathlib import Path
import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(config_name: str) -> dict:
    path = get_project_root() / "configs" / config_name
    if not path.is_file():
        path = Path(__file__).resolve().parents[1] / "configs" / config_name
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value
