"""Tests for dispatch strategies."""

from __future__ import annotations

import pytest

from dispatcher_watts.strategies.threshold import ThresholdStrategy


def test_threshold_decisions() -> None:
    strategy = ThresholdStrategy(charge_below=20.0, discharge_above=50.0)
    prices = [10.0, 20.0, 35.0, 50.0, 80.0]
    decisions = [strategy.decide(i, prices, soc_fraction=0.5) for i in range(len(prices))]
    # <=20 charge, >=50 discharge, in between idle. Boundaries are inclusive.
    assert decisions == [-1.0, -1.0, 0.0, 1.0, 1.0]


def test_threshold_rejects_overlapping_thresholds() -> None:
    with pytest.raises(ValueError, match="must be less than"):
        ThresholdStrategy(charge_below=50.0, discharge_above=20.0)


def test_threshold_name() -> None:
    assert ThresholdStrategy(10.0, 20.0).name == "threshold"
