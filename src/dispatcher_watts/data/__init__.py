"""Market data access layer.

All historical price access goes through the `MarketDataSource` interface
(`base.py`), so the backtest never depends on a concrete provider.
"""
