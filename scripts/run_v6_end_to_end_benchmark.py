#!/usr/bin/env python3
"""Backward compatibility runner for run_nlp_benchmark.py."""
import sys
from pathlib import Path

# Add src to sys.path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from run_nlp_benchmark import main

if __name__ == "__main__":
    main()
