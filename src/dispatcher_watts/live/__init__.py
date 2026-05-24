"""Live paper-trading simulator: deployable strategies run against ERCOT.

The pieces here turn a deployable strategy (``strategies/live.py``) into a
state-persistent paper-trading loop: it observes the market, decides, banks
revenue from both energy and ancillary services, and survives across runs so an
external scheduler can drive it every few minutes.
"""
