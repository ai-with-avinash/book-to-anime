"""Tiny ``.env`` loader.

Avoids a dependency on ``python-dotenv`` for what is a 20-line read.

Format: ``KEY=VALUE`` per line. ``#`` comments and blank lines are ignored.
Surrounding single/double quotes around the value are stripped. Existing
environment variables win, so a real shell ``export`` always overrides
the file — useful for one-shot debugging without editing ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> int:
    """Populate ``os.environ`` from ``path``. Returns the number of keys set.

    Returns 0 (and silently no-ops) if the file does not exist. The caller
    decides whether absence of ``.env`` is an error.
    """

    if not path.is_file():
        return 0
    set_count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            set_count += 1
    return set_count


__all__ = ["load_dotenv"]
