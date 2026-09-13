"""Canonical test paths.

Import these instead of hard-coding ``Path(__file__).parents[n]`` so that test
modules can move between sub-packages without breaking.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

# The historical checkpoint lives in the git-ignored `runs/` tree; tests that
# need it skip when it has not been produced locally.
REFERENCE_DIR = ROOT / "runs" / "diagnose-after"
REFERENCE = REFERENCE_DIR / "last.zip"
