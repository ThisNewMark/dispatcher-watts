"""Smoke tests for the MCP adapter (competition/mcp_server.py).

The service logic is tested in test_competition_service.py; here we only check
the adapter wires the expected tools onto a FastMCP server.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

import polars as pl

from dispatcher_watts.competition.mcp_server import build_server
from dispatcher_watts.competition.service import CompetitionService
from dispatcher_watts.competition.store import CompetitionStore
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, MCPC_SCHEMA, RTM_PRICE_SCHEMA

NOW = dt.datetime(2026, 5, 23, 13, 0, tzinfo=dt.UTC)


class _FakeSource:
    def get_rtm_prices_window(self, start: dt.datetime, end: dt.datetime, hub: str) -> pl.DataFrame:
        starts = [start + dt.timedelta(minutes=15 * i) for i in range(2)]
        return pl.DataFrame(
            {"interval_start": starts, "price": [80.0, 80.0]}, schema=RTM_PRICE_SCHEMA
        ).filter(pl.col("interval_start") < end)

    def get_indicative_mcpc_window(self, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        starts = [start + dt.timedelta(minutes=15 * i) for i in range(2)]
        cols = {f"mcpc_{p}": [0.0] * len(starts) for p in MCPC_PRODUCTS}
        return pl.DataFrame({"interval_start": starts, **cols}, schema=MCPC_SCHEMA).filter(
            pl.col("interval_start") < end
        )


def _server(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = CompetitionStore(tmp_path / "comp.db")
    service = CompetitionService(store, _FakeSource(), data_dir=tmp_path / "live")
    return build_server(service)


def test_server_registers_expected_tools(tmp_path: Path) -> None:
    server = _server(tmp_path)
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "register",
        "submit_decision",
        "get_observation",
        "get_my_state",
        "get_leaderboard",
    }


def test_register_tool_is_callable_through_server(tmp_path: Path) -> None:
    server = _server(tmp_path)
    result = asyncio.run(
        server.call_tool("register", {"display_name": "alice", "email": "a@x.com"})
    )
    # FastMCP returns (content, structured) or content; the structured payload
    # carries the token. Pull whichever the SDK version returns.
    payload = result[1] if isinstance(result, tuple) else result
    assert payload  # a non-empty result came back
