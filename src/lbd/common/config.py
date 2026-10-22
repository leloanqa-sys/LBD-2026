"""Configuration loader."""
import os
import json
from pathlib import Path
from typing import Any, Dict

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent

def load_config(config_name: str) -> Dict[str, Any]:
    root = get_project_root()
    # Try json first or parse basic key-value / yaml
    cfg_path = root / "configs" / config_name
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    # Try importing yaml, else basic parser
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback simple reader for basic types
        result = {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip().strip('"')
                result[k] = v
        return result
