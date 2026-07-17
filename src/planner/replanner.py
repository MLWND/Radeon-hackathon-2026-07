"""
Module 7: Replanner — Deprecated.
Use src/planner/recovery.py which has the full RecoveryManager with
FailureDetector + Replanner. This file is kept only for backward compat.
"""
# Re-export from recovery.py for any old imports
from src.planner.recovery import Replanner  # noqa: F401