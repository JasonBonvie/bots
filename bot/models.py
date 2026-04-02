from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class SignalResult(BaseModel):
    """Complete signal computation result from the momentum bot."""

    timestamp: datetime
    price: float
    price_change_pct: float  # vs 15m ago

    # RSI
    rsi: float

    # EMA
    ema8: float
    ema21: float
    ema_signal: str  # "BULL" or "BEAR"

    # MACD
    macd: float
    macd_signal_line: float

    # Volume
    volume_ratio: float  # current vol / 20-bar rolling mean

    # Close position
    close_position: float  # (close - low) / (high - low)

    # Scores
    score_up: float
    score_dn: float

    # Direction and confidence
    direction: str  # "UP", "DOWN", "SKIP"
    confidence: str  # "HIGH" ≥4, "MEDIUM" ≥3.5, else "SKIP"

    # Filter counts
    filters_passed: int  # count of individual checks that passed
    filters_total: int  # always 5

    # Individual filter results for frontend display
    rsi_filter: str  # "BULL", "BEAR", or "NEUTRAL"
    macd_filter: str  # "BULL" or "BEAR"
    volume_filter: str  # "HIGH" or "NORMAL"
    close_pos_filter: str  # "BULL", "BEAR", or "NEUTRAL"

    # Previous candle info for candle-following strategy
    prev_candle_color: str = "NEUTRAL"  # "GREEN" (bullish), "RED" (bearish), or "NEUTRAL" (doji)


class ConnectionMessage(BaseModel):
    """WebSocket connection status message."""

    type: str = "connection"
    status: str
    message: str


class SignalUpdate(BaseModel):
    """WebSocket signal update message."""

    type: str = "signal"
    data: SignalResult


class ErrorMessage(BaseModel):
    """Error response message."""

    type: str = "error"
    message: str
    retry_after: Optional[int] = None  # seconds until retry


class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    next_candle_close: datetime
    seconds_until_close: int


# ---------------------------------------------------------------------------
# Backtest models
# ---------------------------------------------------------------------------

class BacktestParams(BaseModel):
    """Parameters for a backtest run."""
    start_ts: int                        # Unix timestamp (seconds)
    end_ts: int                          # Unix timestamp (seconds)
    min_score: float = 3.0               # 1–5; threshold to take a trade
    entry_price_cents: int = 50          # assumed entry price in btc_only mode (1-99)
    stake_cents: int = 100               # flat risk per trade in cents
    mode: str = "btc_only"              # "btc_only" | "kalshi"
    slippage_cents: int = 2              # slippage added to entry price (0-10)
    use_limit_order: bool = False        # simulate limit orders instead of market orders
    limit_price_cents: int = 45          # limit order price in cents (1-99)
    limit_window_minutes: int = 3        # minutes after candle open to wait for fill
    # Signal thresholds — match live bot config
    rsi5_bull: float = 55.0             # RSI(5) above = bullish
    rsi5_bear: float = 45.0             # RSI(5) below = bearish
    close_pos_bull: float = 0.65        # close position above = bullish
    close_pos_bear: float = 0.35        # close position below = bearish
    body_ratio_min: float = 0.55        # min body ratio for strong candle
    doji_threshold: float = 0.15        # body ratio below = doji, skip
    volume_ratio_min: float = 1.3       # min volume ratio for confirm
    wick_ratio_min: float = 0.3         # min wick ratio for rejection signal
    consecutive_penalty: int = 4        # consecutive candles before mean reversion penalty
    use_close_position: bool = True      # signal 1
    use_wick_rejection: bool = True      # signal 2
    use_body_strength: bool = True       # signal 3
    use_rsi5: bool = True                # signal 4
    use_volume_confirm: bool = True      # signal 5
    use_engulfing: bool = True           # +1 when candle body fully swallows previous body
    mean_reversion_boost: bool = True    # add +1 to opposite direction on streak (not just penalty)
    require_prev_candle_match: bool = False  # only trade when signal matches prev candle
    # Legacy (kept for API compatibility, not used in scoring)
    require_ema: bool = False
    require_rsi: bool = True
    require_macd: bool = False


class TradeRecord(BaseModel):
    """Single simulated trade in the backtest."""
    time: int                        # Unix timestamp of signal candle close
    direction: str                   # "UP" | "DOWN"
    score_up: float
    score_dn: float
    btc_price: float
    entry_price_cents: int
    exit_price_cents: int            # 100 = win, 0 = loss
    result: str                      # "WIN" | "LOSS"
    pnl_cents: int
    market_ticker: Optional[str] = None  # Kalshi mode only


class FilterStats(BaseModel):
    """How often each individual filter fired across the backtest window."""
    total_candles: int
    ema_bull: int
    ema_bear: int
    rsi_bull: int
    rsi_bear: int
    rsi_neutral: int
    macd_bull: int
    macd_bear: int
    close_pos_bull: int
    close_pos_bear: int
    close_pos_neutral: int
    signals_up: int
    signals_down: int
    signals_skip: int


class EquityPoint(BaseModel):
    """Single point on the equity curve."""
    time: int    # Unix timestamp — matches TradingView line series format
    equity: int  # cumulative P&L in cents


class BacktestResult(BaseModel):
    """Full result of a backtest run."""
    params: BacktestParams
    duration_seconds: float
    total_candles: int
    total_signals: int
    trades_taken: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_cents: int
    avg_pnl_cents: float
    expectancy: float
    max_drawdown_cents: int
    equity_curve: List[EquityPoint]
    trade_log: List[TradeRecord]
    filter_stats: FilterStats
    error: Optional[str] = None
    # Limit order stats (only populated when use_limit_order=True)
    limit_signals: int = 0           # total signals that attempted limit orders
    limit_fills: int = 0             # signals where the limit order was filled
    limit_fill_rate: float = 0.0     # fills / signals
