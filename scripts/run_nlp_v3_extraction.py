#!/usr/bin/env python3
"""Run NLP Extraction (backward compatibility wrapper for run_nlp_extraction.py)."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from run_nlp_extraction import run

if __name__ == "__main__":
    run()
