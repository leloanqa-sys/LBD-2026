"""CLI to validate an artifact file against its contract schema."""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from lbd.common.validation import validate_contract

def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_artifact.py <artifact_json_path> <schema_name>")
        sys.exit(1)
        
    artifact_path = Path(sys.argv[1])
    schema_name = sys.argv[2]
    
    if not artifact_path.exists():
        print(f"Error: Artifact {artifact_path} does not exist.")
        sys.exit(1)
        
    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if isinstance(data, list):
        print(f"Validating list of {len(data)} items...")
        for i, item in enumerate(data):
            valid, msg = validate_contract(item, schema_name)
            if not valid:
                print(f"Item #{i} failed validation: {msg}")
                sys.exit(1)
        print("All items passed validation!")
    else:
        valid, msg = validate_contract(data, schema_name)
        if not valid:
            print(f"Validation failed: {msg}")
            sys.exit(1)
        print("Validation successful!")

if __name__ == "__main__":
    main()
