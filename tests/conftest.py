"""Pytest config for the single-model Space."""
import sys
from pathlib import Path

# Project root must be on sys.path so `import src.*` resolves when pytest
# is run from any directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
