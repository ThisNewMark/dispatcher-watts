"""Direct ERCOT public API client.

The first-party data source. ERCOT publishes every market data product via a
free public REST API at api.ercot.com -- no monthly row cap, no per-call
budget that matters at our cadence. This module replaces the gridstatus.io
middleman for primary fetches; gridstatus stays as a fallback (see the
`CompositeMarketDataSource` in `data/sources.py`).

ERCOT data products: https://www.ercot.com/mp/data-products
API explorer + key signup: https://apiexplorer.ercot.com/
Auth user guide: https://developer.ercot.com/applications/pubapi/

Authentication
--------------
The Public API uses OAuth2 ROPC ("resource owner password credentials") on top
of an Azure API Management subscription key:

  1. POST email + password to the B2C token endpoint.
  2. Receive an `id_token` valid for ~1 hour.
  3. Send the token as `Authorization: Bearer <id_token>` plus the
     subscription key as `Ocp-Apim-Subscription-Key: <key>` on every API call.

Credentials come from environment variables (loaded from .env by the CLI):
``ERCOT_API_USERNAME``, ``ERCOT_API_PASSWORD``, ``ERCOT_API_SUBSCRIPTION_KEY``.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import polars as pl

from dispatcher_watts.data.base import MarketDataSource
from dispatcher_watts.data.schemas import (
    ERCOT_HUBS,
    MCPC_PRODUCTS,
    MCPC_SCHEMA,
    validate_mcpc_frame,
    validate_rtm_frame,
)

# The OAuth2 B2C token endpoint and the public client id are documented in
# ERCOT's auth guide; both are static.
_AUTH_URL: str = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
_AUTH_CLIENT_ID: str = "fec253ea-0d06-4272-a5e6-b478baeecd70"

_BASE_URL: str = "https://api.ercot.com/api/public-reports"

# NP6-905-CD: "Settlement Point Prices at Resource Nodes, Hubs and Load Zones",
# produced every 15 minutes from SCED LMPs.
_SPP_ENDPOINT: str = "/np6-905-cd/spp_node_zone_hub"

# NP6-329-CD: "RTD Indicative Real-Time MCPC". Each RTD run (~every 5 min)
# publishes the projected per-product clearing prices for upcoming 5-minute
# intervals. For live forward-looking decisions, take the most recent
# projection covering each interval. (The 15-minute *settled* MCPC report
# lives at a different endpoint that's still to be located; historical
# backtests use gridstatus's cached settled data instead.)
_INDICATIVE_MCPC_ENDPOINT: str = "/np6-329-cd/rtd_ind_mcpc"

# ERCOT REST timestamps are in Central Prevailing Time (no offset suffix).
_ERCOT_TZ: ZoneInfo = ZoneInfo("America/Chicago")

# How long ERCOT id_tokens are valid. We refresh slightly early to be safe.
_TOKEN_TTL: dt.timedelta = dt.timedelta(minutes=55)
_DEFAULT_PAGE_SIZE: int = 1000
_HTTP_TIMEOUT: float = 60.0


class ErcotDirectSource(MarketDataSource):
    """`MarketDataSource` implementation backed by ERCOT's public REST API."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        subscription_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Create a source.

        Args:
            username: ERCOT developer-portal email. Falls back to env
                ``ERCOT_API_USERNAME``.
            password: ERCOT account password. Falls back to env
                ``ERCOT_API_PASSWORD``.
            subscription_key: APIM primary subscription key. Falls back to env
                ``ERCOT_API_SUBSCRIPTION_KEY``.
            client: a pre-built httpx client (primarily for tests).
        """
        self._username = username
        self._password = password
        self._subscription_key = subscription_key
        self._client: httpx.Client | None = client
        self._token: str | None = None
        self._token_expiry: dt.datetime | None = None

    # --- transport ---------------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=_HTTP_TIMEOUT)
        return self._client

    def _credentials(self) -> tuple[str, str, str]:
        username = self._username or os.environ.get("ERCOT_API_USERNAME")
        password = self._password or os.environ.get("ERCOT_API_PASSWORD")
        sub_key = self._subscription_key or os.environ.get("ERCOT_API_SUBSCRIPTION_KEY")
        missing = [
            name
            for name, value in (
                ("ERCOT_API_USERNAME", username),
                ("ERCOT_API_PASSWORD", password),
                ("ERCOT_API_SUBSCRIPTION_KEY", sub_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"ERCOT credentials missing: {', '.join(missing)}. "
                f"Set them in .env or pass to the constructor."
            )
        # The asserts narrow the optional types after the missing-check above.
        assert username is not None and password is not None and sub_key is not None
        return username, password, sub_key

    def _get_token(self) -> str:
        """Return a valid OAuth2 id_token, refreshing if expired or absent."""
        now = dt.datetime.now(dt.UTC)
        if self._token and self._token_expiry and now < self._token_expiry:
            return self._token
        username, password, _ = self._credentials()
        response = self._get_client().post(
            _AUTH_URL,
            data={
                "username": username,
                "password": password,
                "grant_type": "password",
                "scope": f"openid {_AUTH_CLIENT_ID} offline_access",
                "client_id": _AUTH_CLIENT_ID,
                "response_type": "id_token",
            },
        )
        response.raise_for_status()
        body = response.json()
        token = body.get("id_token")
        if not token:
            raise RuntimeError(f"ERCOT auth response had no id_token; body keys: {list(body)}")
        self._token = token
        self._token_expiry = now + _TOKEN_TTL
        return token

    def _api_get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Authenticated GET with retry on 401 (auth refresh) and 429 (backoff).

        ERCOT enforces a per-minute API rate limit in addition to the monthly
        budget. On 429 we wait the time the server hints at via Retry-After
        (or a default), then try again -- a small number of times.
        """
        _, _, sub_key = self._credentials()
        for attempt in range(1, 5):
            response = self._get_client().get(
                _BASE_URL + endpoint,
                headers={
                    "Authorization": f"Bearer {self._get_token()}",
                    "Ocp-Apim-Subscription-Key": sub_key,
                },
                params=params,
            )
            if response.status_code == 401 and attempt < 4:
                # Token may have aged out faster than expected; force a refresh.
                self._token = None
                continue
            if response.status_code == 429 and attempt < 4:
                # ERCOT may send Retry-After in seconds; default to 30s if absent.
                wait_s = int(response.headers.get("Retry-After", "30"))
                time.sleep(min(wait_s, 60))
                continue
            if response.status_code == 403 and attempt < 3:
                # A transient gateway rejection clears on a short retry. A
                # persistent 403 (e.g. a geo-blocked or non-US exit IP, common
                # behind a VPN) won't, and falls through to surface clearly.
                time.sleep(3)
                continue
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data
        raise RuntimeError(f"ERCOT API exhausted retries on {endpoint}")

    def _paginated_api_get(
        self,
        endpoint: str,
        base_params: dict[str, Any],
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int = 1000,
    ) -> tuple[list[str], list[list[Any]]]:
        """Walk every page; return the field-name list and concatenated rows.

        ERCOT public API responses look like::

            {"fields": ["deliveryDate", "deliveryHour", ...],
             "data":   [[...row...], [...row...], ...],
             "_meta":  {"totalPages": N, ...}}

        The "fields" array names the positions in each "data" row.
        """
        all_rows: list[list[Any]] = []
        fields: list[str] = []
        page = 1
        while True:
            params = {**base_params, "page": page, "size": page_size}
            body = self._api_get(endpoint, params)
            if not fields:
                fields = [
                    f.get("name", f) if isinstance(f, dict) else f for f in body.get("fields", [])
                ]
            all_rows.extend(body.get("data", []))
            meta = body.get("_meta", {})
            total_pages = int(meta.get("totalPages", 1) or 1)
            if page >= total_pages or page >= max_pages or not body.get("data"):
                break
            page += 1
        return fields, all_rows

    # --- MarketDataSource interface ---------------------------------------

    def get_rtm_prices(self, year: int, hub: str) -> pl.DataFrame:
        if hub not in ERCOT_HUBS:
            raise ValueError(f"unknown ERCOT hub {hub!r}; expected one of {ERCOT_HUBS}")
        fields, rows = self._paginated_api_get(
            _SPP_ENDPOINT,
            {
                "deliveryDateFrom": f"{year}-01-01",
                "deliveryDateTo": f"{year}-12-31",
                "settlementPoint": hub,
            },
        )
        return normalize_ercot_spp_response(fields, rows)

    def get_rtm_prices_window(
        self,
        start: dt.datetime,
        end: dt.datetime,
        hub: str,
    ) -> pl.DataFrame:
        """Return RT settlement prices for `hub` with interval in ``[start, end)``.

        The windowed counterpart of `get_rtm_prices`, for the live loop: fetching
        a whole calendar year every few minutes would be wasteful and rate-limit
        prone. The SPP endpoint filters by *delivery date* (CPT, day-granular),
        so we query the day range covering the window and slice to the exact
        UTC interval afterwards.
        """
        if hub not in ERCOT_HUBS:
            raise ValueError(f"unknown ERCOT hub {hub!r}; expected one of {ERCOT_HUBS}")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware datetimes")
        fields, rows = self._paginated_api_get(
            _SPP_ENDPOINT,
            {
                "deliveryDateFrom": start.astimezone(_ERCOT_TZ).date().isoformat(),
                "deliveryDateTo": end.astimezone(_ERCOT_TZ).date().isoformat(),
                "settlementPoint": hub,
            },
            max_pages=50,
        )
        frame = normalize_ercot_spp_response(fields, rows)
        return frame.filter((pl.col("interval_start") >= start) & (pl.col("interval_start") < end))

    def get_rtm_mcpc(self, year: int) -> pl.DataFrame:
        # ERCOT publishes 5-minute *indicative* MCPC at the rtd_ind_mcpc
        # endpoint, but the 15-minute *settled* MCPC report (which is what we
        # cache historically) lives at a different report we haven't located
        # yet. Until we find it, MCPC historical fetches fall back to
        # gridstatus via the composite source. Live 5-min indicative
        # available via get_indicative_mcpc_window().
        raise NotImplementedError(
            "ercot-direct 15-min settled MCPC fetch is not yet wired; "
            "for historical MCPC use gridstatus, for live forward MCPC use "
            "get_indicative_mcpc_window()"
        )

    def get_indicative_mcpc_window(
        self,
        start: dt.datetime,
        end: dt.datetime,
    ) -> pl.DataFrame:
        """Return latest-projection 5-min MCPCs for intervals ending in [start, end).

        For each 5-minute interval whose ending falls inside the window, picks
        the row from the most recent RTD run that projected that interval
        (i.e. the most up-to-date cleared price). Returns a wide-format frame
        conforming to ``MCPC_SCHEMA``, with ``interval_start`` = 5 minutes
        before ``intervalEnding``.

        Use this for **live** decisions; for historical settled MCPCs use the
        gridstatus-cached data.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware datetimes")
        # ERCOT's filter takes naive CPT timestamps in ISO format.
        start_cpt = start.astimezone(_ERCOT_TZ).strftime("%Y-%m-%dT%H:%M:%S")
        end_cpt = end.astimezone(_ERCOT_TZ).strftime("%Y-%m-%dT%H:%M:%S")
        fields, rows = self._paginated_api_get(
            _INDICATIVE_MCPC_ENDPOINT,
            {
                "intervalEndingFrom": start_cpt,
                "intervalEndingTo": end_cpt,
            },
            max_pages=20,
        )
        return normalize_ercot_indicative_mcpc_response(fields, rows)


def normalize_ercot_spp_response(fields: list[str], rows: list[list[Any]]) -> pl.DataFrame:
    """Convert an ERCOT NP6-905-CD response to a frame matching RTM_PRICE_SCHEMA.

    The report's row schema is well-known:
    ``deliveryDate`` (YYYY-MM-DD), ``deliveryHour`` (1-24 in CPT),
    ``deliveryInterval`` (1-4 within the hour), ``settlementPointName``,
    ``settlementPointType``, ``settlementPointPrice`` ($/MWh), ``DSTFlag``.

    We compose ``interval_start`` from (date, hour, interval) in
    America/Chicago and convert to UTC -- ERCOT publishes in Central Prevailing
    Time. ``DSTFlag`` (``Y`` on the fall-back interval) is honored by polars'
    "ambiguous=earliest" / "latest" tag inside the raw data.
    """
    if not rows:
        return pl.DataFrame(
            schema={
                "interval_start": pl.Datetime(time_unit="us", time_zone="UTC"),
                "price": pl.Float64,
            }
        )

    raw = pl.DataFrame(rows, schema=fields, orient="row")
    # ERCOT hours are 1-24 (hour-ending in some reports; hour-beginning in
    # others). NP6-905-CD documents hour 1 == 00:00-01:00 (hour-beginning),
    # so converting (hour - 1) gives the interval-start hour.
    # Each hour has 4 intervals of 15 minutes; interval 1 starts at minute 0.
    normalized = raw.select(
        local_naive=(
            pl.col("deliveryDate").str.to_datetime("%Y-%m-%d", time_unit="us")
            + pl.duration(hours=pl.col("deliveryHour").cast(pl.Int64) - 1)
            + pl.duration(minutes=(pl.col("deliveryInterval").cast(pl.Int64) - 1) * 15)
        ),
        price=pl.col("settlementPointPrice").cast(pl.Float64),
        dst_flag=pl.col("DSTFlag"),
    )
    # Localize to America/Chicago, then convert to UTC. ERCOT's DST flag
    # tells us which side of the fall-back ambiguous hour each row belongs
    # to: the flag is "set" on rows in the repeated hour (the later instance).
    # The public API sends DSTFlag as bool; some other ERCOT endpoints send
    # "Y"/"N" strings. Cast to string and accept either form so this is
    # robust across endpoints.
    dst_truthy = pl.col("dst_flag").cast(pl.Utf8).is_in(("Y", "true", "True"))
    normalized = normalized.with_columns(
        interval_start=pl.col("local_naive")
        .dt.replace_time_zone(
            "America/Chicago",
            ambiguous=pl.when(dst_truthy).then(pl.lit("latest")).otherwise(pl.lit("earliest")),
            non_existent="null",
        )
        .dt.convert_time_zone("UTC")
    ).select(
        pl.col("interval_start"),
        pl.col("price"),
    )
    normalized = normalized.drop_nulls("interval_start").sort("interval_start")
    return validate_rtm_frame(normalized)


def normalize_ercot_indicative_mcpc_response(
    fields: list[str], rows: list[list[Any]]
) -> pl.DataFrame:
    """Convert NP6-329-CD response to a frame matching MCPC_SCHEMA.

    The endpoint returns multiple projections per interval (one per RTD run
    that reached that interval). We keep only the latest projection per
    intervalEnding, then map ``intervalEnding - 5 min`` (in America/Chicago)
    to UTC ``interval_start``.
    """
    if not rows:
        return pl.DataFrame(schema=MCPC_SCHEMA)
    raw = pl.DataFrame(rows, schema=fields, orient="row")
    # Pick the row with the latest RTDTimestamp for each intervalEnding.
    raw = (
        raw.with_columns(
            pl.col("RTDTimestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="us"),
            pl.col("intervalEnding").str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="us"),
        )
        .sort("RTDTimestamp", descending=True)
        .unique(subset=["intervalEnding"], keep="first")
    )
    dst_truthy = pl.col("intervalRepeatHourFlag").cast(pl.Utf8).is_in(("Y", "true", "True"))
    out = (
        raw.with_columns(
            interval_start=(pl.col("intervalEnding") - pl.duration(minutes=5))
            .dt.replace_time_zone(
                "America/Chicago",
                ambiguous=pl.when(dst_truthy).then(pl.lit("latest")).otherwise(pl.lit("earliest")),
                non_existent="null",
            )
            .dt.convert_time_zone("UTC"),
            mcpc_regup=pl.col("REGUP").cast(pl.Float64),
            mcpc_regdn=pl.col("REGDN").cast(pl.Float64),
            mcpc_rrs=pl.col("RRS").cast(pl.Float64),
            mcpc_ecrs=pl.col("ECRS").cast(pl.Float64),
            mcpc_nspin=pl.col("NSPIN").cast(pl.Float64),
        )
        .select(["interval_start", *(f"mcpc_{p}" for p in MCPC_PRODUCTS)])
        .drop_nulls("interval_start")
        .sort("interval_start")
    )
    return validate_mcpc_frame(out)


__all__ = [
    "ErcotDirectSource",
    "normalize_ercot_indicative_mcpc_response",
    "normalize_ercot_spp_response",
]
