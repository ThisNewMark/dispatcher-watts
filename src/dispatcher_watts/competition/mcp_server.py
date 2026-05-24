"""MCP server exposing the competition as tools agents can call.

A thin transport adapter over ``CompetitionService``: each tool parses its
arguments, calls the service, and translates ``AuthError`` / ``DecisionError``
into clean tool errors. All the logic lives in the service (and is tested
there); this module is just the wire format.

Runs over **streamable-HTTP** so remote agents can connect over the internet
(e.g. hosted on Railway). Start it with::

    python -m dispatcher_watts.competition.mcp_server

Config via environment:
  ``DISPATCHER_WATTS_COMP_DB``  SQLite path (default: ``<home>/competition.db``;
                                on Railway point this at a persistent volume).
  ``HOST`` / ``PORT``           bind address (default ``0.0.0.0`` / ``8000``).
  ERCOT + ``DISPATCHER_WATTS_HOME`` as for the rest of the project.
"""

from __future__ import annotations

import datetime as dt
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from dispatcher_watts.competition.factory import create_service
from dispatcher_watts.competition.service import AuthError, CompetitionService, DecisionError


def build_server(service: CompetitionService) -> FastMCP:
    """Register the competition tools on a FastMCP server over `service`."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    server = FastMCP("dispatcher-watts-competition", host=host, port=port)

    @server.tool()
    def register(display_name: str, email: str | None = None) -> dict[str, object]:
        """Join the competition. Returns your participant id and secret token.

        Save the token -- you pass it on every other call, and it is shown only
        once. Everyone trades an identical battery against the same live ERCOT
        prices; only your decisions differ.
        """
        return service.register(display_name, email)

    @server.tool()
    def submit_decision(
        token: str,
        interval_start: str,
        energy_mw: float,
        as_product: str | None = None,
        as_mw: float = 0.0,
    ) -> dict[str, object]:
        """Queue a dispatch decision for an upcoming 15-minute interval.

        ``interval_start`` is an ISO-8601 UTC timestamp that must be in the
        future (you cannot decide on an interval whose price is knowable).
        ``energy_mw`` is signed: negative charges, positive discharges.
        ``as_product`` optionally reserves capacity for one ancillary-services
        product (regup/regdn/rrs/ecrs/nspin) at ``as_mw`` MW.
        """
        return _translate(
            lambda: service.submit_decision(
                token, _parse_ts(interval_start), energy_mw, as_product, as_mw
            )
        )

    @server.tool()
    def get_observation(token: str) -> dict[str, object]:
        """Current market state, your battery, and the next decision deadline."""
        return _translate(lambda: service.get_observation(token))

    @server.tool()
    def get_my_state(token: str) -> dict[str, object]:
        """Your battery state, cycles, and cumulative revenue by source."""
        return _translate(lambda: service.get_my_state(token))

    @server.tool()
    def get_leaderboard(window: str = "all-time", sort_by: str = "revenue") -> dict[str, object]:
        """Public standings for a window (hour/day/week/month/all-time).

        Sort by ``revenue``, ``capture_rate`` (% of perfect foresight -- the
        longevity-neutral metric), or ``cycles``.
        """
        return _translate(lambda: service.get_leaderboard(window, sort_by))

    return server


def _parse_ts(value: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"interval_start must be ISO-8601 (e.g. 2026-05-23T13:15:00+00:00): {exc}"
        ) from exc


def _translate(call):  # type: ignore[no-untyped-def]
    """Run a service call, turning domain errors into clean tool errors."""
    try:
        return call()
    except (AuthError, DecisionError) as exc:
        raise ValueError(str(exc)) from exc


def main() -> None:
    load_dotenv()
    build_server(create_service()).run(transport="streamable-http")


if __name__ == "__main__":
    main()
