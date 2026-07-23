"""Configuration loading for JSON-compatible YAML files."""
from __future__ import annotations
from pathlib import Path
import json
try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

def load_yaml(path:str|Path)->dict:
    """Load a configuration file; JSON syntax is supported without PyYAML."""
    text=Path(path).read_text(encoding='utf-8')
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return json.loads(text)
