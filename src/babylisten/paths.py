"""Data directory. Per-user settings live in $BABYLISTEN_HOME (default ~/.babylisten)."""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("BABYLISTEN_HOME", "~/.babylisten")).expanduser()
CONFIG = HOME / "config.json"
