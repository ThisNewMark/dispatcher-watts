"""Build a configured CompetitionService from environment/defaults.

Kept separate from ``mcp_server`` so the heartbeat CLI (and tests) can construct
a service without importing the MCP SDK.
"""

from __future__ import annotations

import os
from pathlib import Path

from dispatcher_watts.competition.service import CompetitionService
from dispatcher_watts.competition.store import CompetitionStore
from dispatcher_watts.data.ercot_direct import ErcotDirectSource
from dispatcher_watts.paths import home_dir

COMP_DB_ENV_VAR = "DISPATCHER_WATTS_COMP_DB"


def competition_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve the SQLite path: arg > env > ``<home>/competition.db``."""
    if db_path is not None:
        return Path(db_path)
    override = os.environ.get(COMP_DB_ENV_VAR)
    return Path(override) if override else home_dir() / "competition.db"


def create_service(db_path: Path | str | None = None) -> CompetitionService:
    """A service backed by SQLite + the direct ERCOT client."""
    store = CompetitionStore(competition_db_path(db_path))
    return CompetitionService(store, ErcotDirectSource())


__all__ = ["COMP_DB_ENV_VAR", "competition_db_path", "create_service"]
