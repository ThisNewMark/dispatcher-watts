"""Tests for the direct ERCOT client (data/ercot_direct.py).

Transport (auth/retry/pagination) is exercised with a fake httpx client built
from real ``httpx.Response`` objects, so status codes, headers, and
``raise_for_status`` behave exactly as in production. The response normalizers
are pure and tested directly.
"""

from __future__ import annotations

import datetime as dt

import httpx
import pytest

from dispatcher_watts.data.ercot_direct import (
    ErcotDirectSource,
    normalize_ercot_indicative_mcpc_response,
    normalize_ercot_spp_response,
)
from dispatcher_watts.data.schemas import MCPC_SCHEMA, RTM_PRICE_SCHEMA

_REQ = httpx.Request("GET", "https://api.ercot.com/x")


def _token_response() -> httpx.Response:
    return httpx.Response(200, json={"id_token": "tok"}, request=_REQ)


def _page(fields: list[str], data: list[list[object]], total_pages: int = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json={"fields": fields, "data": data, "_meta": {"totalPages": total_pages}},
        request=_REQ,
    )


class _FakeClient:
    """Stand-in for httpx.Client: serves a token on POST and a GET queue."""

    def __init__(self, get_responses: list[httpx.Response]) -> None:
        self._get_responses = list(get_responses)
        self.post_calls = 0
        self.get_params: list[dict[str, object]] = []

    def post(self, url: str, data: dict[str, object] | None = None) -> httpx.Response:
        self.post_calls += 1
        return _token_response()

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, object] | None = None,
    ) -> httpx.Response:
        self.get_params.append(params or {})
        return self._get_responses.pop(0)


_SPP_FIELDS = [
    "deliveryDate",
    "deliveryHour",
    "deliveryInterval",
    "settlementPointName",
    "settlementPointType",
    "settlementPointPrice",
    "DSTFlag",
]


def _spp_row(
    date: str, hour: int, interval: int, price: float, dst: object = False
) -> list[object]:
    return [date, hour, interval, "HB_HOUSTON", "HU", price, dst]


def _source(client: _FakeClient) -> ErcotDirectSource:
    return ErcotDirectSource(username="u", password="p", subscription_key="k", client=client)


# --- auth + transport --------------------------------------------------------


def test_token_is_fetched_once_and_reused() -> None:
    client = _FakeClient(
        [
            _page(_SPP_FIELDS, [_spp_row("2025-01-01", 1, 1, 25.0)]),
            _page(_SPP_FIELDS, [_spp_row("2025-01-01", 1, 2, 26.0)]),
        ]
    )
    source = _source(client)
    source.get_rtm_prices(2025, "HB_HOUSTON")
    source.get_rtm_prices(2025, "HB_HOUSTON")
    assert client.post_calls == 1  # token cached across both API calls


def test_401_forces_token_refresh_and_retries() -> None:
    client = _FakeClient(
        [
            httpx.Response(401, request=_REQ),
            _page(_SPP_FIELDS, [_spp_row("2025-01-01", 1, 1, 25.0)]),
        ]
    )
    source = _source(client)
    df = source.get_rtm_prices(2025, "HB_HOUSTON")
    assert df.height == 1
    assert client.post_calls == 2  # initial token + refresh after 401


def test_429_backs_off_then_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[int] = []
    monkeypatch.setattr("dispatcher_watts.data.ercot_direct.time.sleep", lambda s: slept.append(s))
    client = _FakeClient(
        [
            httpx.Response(429, headers={"Retry-After": "7"}, request=_REQ),
            _page(_SPP_FIELDS, [_spp_row("2025-01-01", 1, 1, 25.0)]),
        ]
    )
    df = _source(client).get_rtm_prices(2025, "HB_HOUSTON")
    assert df.height == 1
    assert slept == [7]  # honored Retry-After


def test_403_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # A transient 403 should be retried after a short backoff, not surfaced.
    monkeypatch.setattr("dispatcher_watts.data.ercot_direct.time.sleep", lambda s: None)
    client = _FakeClient(
        [
            httpx.Response(403, request=_REQ),
            _page(_SPP_FIELDS, [_spp_row("2025-01-01", 1, 1, 25.0)]),
        ]
    )
    df = _source(client).get_rtm_prices(2025, "HB_HOUSTON")
    assert df.height == 1


def test_persistent_403_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    # A persistent 403 (e.g. a geo-blocked VPN exit IP) exhausts the brief
    # retry and surfaces as an HTTP error rather than looping forever.
    monkeypatch.setattr("dispatcher_watts.data.ercot_direct.time.sleep", lambda s: None)
    client = _FakeClient([httpx.Response(403, request=_REQ) for _ in range(3)])
    with pytest.raises(httpx.HTTPStatusError):
        _source(client).get_rtm_prices(2025, "HB_HOUSTON")


def test_pagination_concatenates_pages() -> None:
    client = _FakeClient(
        [
            _page(_SPP_FIELDS, [_spp_row("2025-01-01", 1, 1, 25.0)], total_pages=2),
            _page(_SPP_FIELDS, [_spp_row("2025-01-01", 1, 2, 26.0)], total_pages=2),
        ]
    )
    df = _source(client).get_rtm_prices(2025, "HB_HOUSTON")
    assert df.height == 2
    assert df["price"].to_list() == [25.0, 26.0]


def test_missing_credentials_raises() -> None:
    source = ErcotDirectSource(username=None, password=None, subscription_key=None, client=None)
    # No env vars in test; credential check should fire before any network use.
    with pytest.raises(RuntimeError, match="ERCOT credentials missing"):
        source._credentials()


def test_unknown_hub_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ERCOT hub"):
        _source(_FakeClient([])).get_rtm_prices(2025, "HB_NOWHERE")


def test_get_rtm_mcpc_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        _source(_FakeClient([])).get_rtm_mcpc(2025)


# --- windowed price fetch ----------------------------------------------------


def test_prices_window_slices_to_range_and_passes_dates() -> None:
    client = _FakeClient(
        [
            _page(
                _SPP_FIELDS,
                [
                    _spp_row("2025-01-01", 1, 1, 25.0),  # 06:00 UTC
                    _spp_row("2025-01-01", 1, 2, 26.0),  # 06:15 UTC
                    _spp_row("2025-01-01", 1, 3, 27.0),  # 06:30 UTC
                ],
            )
        ]
    )
    source = _source(client)
    start = dt.datetime(2025, 1, 1, 6, 15, tzinfo=dt.UTC)
    end = dt.datetime(2025, 1, 1, 6, 30, tzinfo=dt.UTC)
    df = source.get_rtm_prices_window(start, end, "HB_HOUSTON")
    # Half-open [start, end): only the 06:15 row.
    assert df["price"].to_list() == [26.0]
    assert client.get_params[0]["settlementPoint"] == "HB_HOUSTON"


def test_prices_window_rejects_naive_datetimes() -> None:
    naive = dt.datetime(2025, 1, 1, 6, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        _source(_FakeClient([])).get_rtm_prices_window(naive, naive, "HB_HOUSTON")


def test_indicative_mcpc_window_rejects_naive_datetimes() -> None:
    naive = dt.datetime(2025, 1, 1, 6, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        _source(_FakeClient([])).get_indicative_mcpc_window(naive, naive)


# --- normalizers -------------------------------------------------------------


def test_normalize_spp_hour_beginning_to_utc() -> None:
    # ERCOT hour 1 / interval 1 == 00:00 CPT; January is CST (UTC-6) -> 06:00 UTC.
    out = normalize_ercot_spp_response(_SPP_FIELDS, [_spp_row("2025-01-01", 1, 1, 25.0)])
    assert out.schema == RTM_PRICE_SCHEMA
    assert out["interval_start"].to_list() == [dt.datetime(2025, 1, 1, 6, 0, tzinfo=dt.UTC)]


def test_normalize_spp_accepts_string_dst_flag() -> None:
    # Some ERCOT endpoints send DSTFlag as "Y"/"N" rather than bool; both work.
    out = normalize_ercot_spp_response(_SPP_FIELDS, [_spp_row("2025-06-01", 5, 2, 40.0, dst="N")])
    assert out.height == 1
    assert out["price"].to_list() == [40.0]


def test_normalize_spp_empty_returns_typed_empty_frame() -> None:
    out = normalize_ercot_spp_response(_SPP_FIELDS, [])
    assert out.is_empty()
    assert out.columns == ["interval_start", "price"]


_MCPC_FIELDS = [
    "RTDTimestamp",
    "intervalEnding",
    "REGUP",
    "REGDN",
    "RRS",
    "ECRS",
    "NSPIN",
    "intervalRepeatHourFlag",
]


def test_normalize_indicative_mcpc_keeps_latest_projection() -> None:
    rows = [
        # Same interval, two RTD runs: the later run's REGUP must win.
        ["2025-01-01T00:00:00", "2025-01-01T00:05:00", 5.0, 1.0, 2.0, 3.0, 4.0, False],
        ["2025-01-01T00:04:00", "2025-01-01T00:05:00", 9.0, 1.0, 2.0, 3.0, 4.0, False],
    ]
    out = normalize_ercot_indicative_mcpc_response(_MCPC_FIELDS, rows)
    assert out.schema == MCPC_SCHEMA
    assert out.height == 1
    assert out["mcpc_regup"].to_list() == [9.0]
    # interval_start = intervalEnding - 5min = 00:00 CPT -> 06:00 UTC.
    assert out["interval_start"].to_list() == [dt.datetime(2025, 1, 1, 6, 0, tzinfo=dt.UTC)]


def test_normalize_indicative_mcpc_empty_returns_typed_empty_frame() -> None:
    out = normalize_ercot_indicative_mcpc_response(_MCPC_FIELDS, [])
    assert out.is_empty()
    assert out.schema == MCPC_SCHEMA
