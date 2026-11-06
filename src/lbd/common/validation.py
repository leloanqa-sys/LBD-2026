"""JSON Schema validation for Data Contracts A through E."""
import json
from pathlib import Path
from typing import Any, Dict, Tuple

def validate_contract(data: Dict[str, Any], schema_name: str) -> Tuple[bool, str]:
    root = Path(__file__).resolve().parent.parent.parent.parent
    schema_path = Path(__file__).resolve().parents[1] / "contracts" / schema_name
    if not schema_path.exists():
        return False, f"Schema file {schema_name} not found at {schema_path}"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
        return True, "Validation successful."
    except ImportError:
        return False, "jsonschema is required; validation was not performed."
    except Exception as e:
        return False, f"Contract validation error: {str(e)}"
