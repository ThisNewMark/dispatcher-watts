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
from typing import Any

import httpx
import polars as pl

from dispatcher_watts.data.base import MarketDataSource
from dispatcher_watts.data.schemas import ERCOT_HUBS, validate_rtm_frame

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
        """Authenticated GET against `_BASE_URL + endpoint` with retry on 401."""
        _, _, sub_key = self._credentials()
        for attempt in (1, 2):
            response = self._get_client().get(
                _BASE_URL + endpoint,
                headers={
                    "Authorization": f"Bearer {self._get_token()}",
                    "Ocp-Apim-Subscription-Key": sub_key,
                },
                params=params,
            )
            if response.status_code == 401 and attempt == 1:
                # Token may have aged out faster than expected; force a refresh.
                self._token = None
                continue
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data
        raise RuntimeError("unreachable: api_get loop did not return")

    def _paginated_api_get(
        self,
        endpoint: str,
        base_params: dict[str, Any],
        page_size: int = _DEFAULT_PAGE_SIZE,
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
            if page >= total_pages or not body.get("data"):
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

    def get_rtm_mcpc(self, year: int) -> pl.DataFrame:
        # The post-RTC+B real-time MCPC endpoint name needs verification
        # against the live API; deliberately not implemented in this first
        # pass so the composite source falls back to gridstatus for MCPC.
        raise NotImplementedError(
            "ercot-direct MCPC fetch not yet wired -- pending endpoint probe; "
            "for now MCPC falls back to gridstatus.io via CompositeMarketDataSource"
        )


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
            ambiguous=pl.when(dst_truthy)
            .then(pl.lit("latest"))
            .otherwise(pl.lit("earliest")),
            non_existent="null",
        )
        .dt.convert_time_zone("UTC")
    ).select(
        pl.col("interval_start"),
        pl.col("price"),
    )
    normalized = normalized.drop_nulls("interval_start").sort("interval_start")
    return validate_rtm_frame(normalized)


__all__ = ["ErcotDirectSource", "normalize_ercot_spp_response"]
