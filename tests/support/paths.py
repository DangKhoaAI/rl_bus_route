"""Canonical test paths.

Import these instead of hard-coding ``Path(__file__).parents[n]`` so that test
modules can move between sub-packages without breaking.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

# Frozen test data lives outside `tests/` so that `tests/` holds only test code.
DATA = ROOT / "fixtures"
GOLDEN = DATA / "golden"
REFERENCE_DIR = DATA / "reference"
REFERENCE = REFERENCE_DIR / "diagnose-after" / "last.zip"
REFERENCE_JSON = REFERENCE_DIR / "reference.json"
