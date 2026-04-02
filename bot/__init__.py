"""
BTC/Kalshi Momentum Bot Module

This module provides:
- signals: Compute momentum signals from BTC price data
- scheduler: Sync signal computation to 15-minute candle closes
- models: Pydantic models for API responses
"""

from .models import SignalResult, ConnectionMessage, SignalUpdate, ErrorMessage, HealthCheck
from .signals import compute_signals, get_individual_signal_states
from .scheduler import scheduler, SignalScheduler

__all__ = [
    "SignalResult",
    "ConnectionMessage",
    "SignalUpdate",
    "ErrorMessage",
    "HealthCheck",
    "compute_signals",
    "get_individual_signal_states",
    "scheduler",
    "SignalScheduler",
]
