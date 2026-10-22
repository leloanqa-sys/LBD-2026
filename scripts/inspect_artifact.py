"""CLI to inspect summary statistics of a generated artifact."""
import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_artifact.py <artifact_path>")
        sys.exit(1)
        
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)
        
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"--- Artifact Summary: {p.name} ---")
    if isinstance(data, list):
        print(f"Type: List, Total Elements: {len(data)}")
        if data:
            print("Keys in first element:", list(data[0].keys()))
    elif isinstance(data, dict):
        print("Type: Dict, Keys:", list(data.keys()))
        for k, v in data.items():
            if isinstance(v, list):
                print(f"  - {k}: {len(v)} items")
            else:
                print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
