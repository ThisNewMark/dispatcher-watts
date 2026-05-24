"""Resolve stable on-disk locations independent of the process working directory.

The live simulator runs unattended (cron, launchd, a systemd timer), and those
launchers rarely set the working directory to the project root. Anchoring the
state and capture-log paths to a discovered *home* directory -- rather than to
``Path("state")`` relative to wherever the process happened to start -- prevents
the loop from silently forking its state when launched from elsewhere.

Resolution order for the home directory:

  1. the ``DISPATCHER_WATTS_HOME`` environment variable, if set;
  2. the repository root (nearest ancestor of this file that contains a
     ``pyproject.toml``) -- the normal case for a source checkout;
  3. the current working directory, as a last resort.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME_ENV_VAR: str = "DISPATCHER_WATTS_HOME"


def home_dir() -> Path:
    """Return the stable base directory for state and captured data."""
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    repo_root = _find_repo_root()
    return repo_root if repo_root is not None else Path.cwd()


def _find_repo_root() -> Path | None:
    """Nearest ancestor of this module that contains a ``pyproject.toml``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


__all__ = ["HOME_ENV_VAR", "home_dir"]
