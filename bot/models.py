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
    start_ts: int                    # Unix timestamp (seconds)
    end_ts: int                      # Unix timestamp (seconds)
    min_score: float = 3.0           # 1.0–4.0; threshold to take a trade
    entry_price_cents: int = 50      # assumed entry price in btc_only mode (1-99)
    stake_cents: int = 100           # flat risk per trade in cents
    mode: str = "btc_only"           # "btc_only" | "kalshi"
    require_ema: bool = True
    require_rsi: bool = True
    require_macd: bool = True
    require_prev_candle_match: bool = False  # only trade when signal matches previous candle color
    slippage_cents: int = 2          # slippage added to entry price (worsens fill, 0-10)
    use_limit_order: bool = False    # simulate limit orders instead of market orders
    limit_price_cents: int = 45      # limit order price in cents (1-99)
    limit_window_minutes: int = 3    # how many minutes after candle open to wait for fill


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
