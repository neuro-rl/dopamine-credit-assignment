"""Puts model/ on sys.path so tests can use the same flat-import style as the
rest of the codebase (e.g. `from arm import SO101PlanarArm`, not
`from model.arm import ...`) — matches evsim/encoders' existing convention.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
