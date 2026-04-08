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
    prev_candle_color: str = "NEUTRAL"   # color of df.iloc[-3] (one before the signal candle)
    signal_candle_color: str = "NEUTRAL" # color of df.iloc[-2] (the candle that JUST closed)

    # New context signal outputs
    mtf_bias: str = "NEUTRAL"    # "BULL", "BEAR", or "NEUTRAL" — combined 1h/4h EMA context
    vwap: float = 0.0            # session VWAP price (0.0 if not computed)
    vwap_dev_pct: float = 0.0    # (close - vwap) / vwap * 100
    funding_rate: float = 0.0    # perpetual funding rate (0.0 if not fetched)


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
# Shared order types
# ---------------------------------------------------------------------------

class LimitTier(BaseModel):
    """One rung of a limit order ladder — fills if price dips to price_cents within window."""
    price_cents: int = 40           # limit price for this tier (must be < primary limit)
    window_minutes: int = 5         # cancel if unfilled after this many candles
    stake_dollars: float = 1.0      # dollar amount to deploy at this price level


# ---------------------------------------------------------------------------
# Backtest models
# ---------------------------------------------------------------------------

class BacktestParams(BaseModel):
    """Parameters for a backtest run."""
    start_ts: int                        # Unix timestamp (seconds)
    end_ts: int                          # Unix timestamp (seconds)
    strategy_mode: str = "signals"       # "signals" | "follow_candle"
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
    # Martingale simulation
    use_martingale: bool = False             # double dollar stake after each loss
    martingale_multiplier: float = 2.0       # stake multiplier per loss level
    max_martingale_level: int = 5            # stop growing after this many consecutive losses
    require_prev_candle_match: bool = False  # only trade when signal matches prev candle
    # Limit order ladder (additional tiers below primary limit)
    limit_ladder: List[LimitTier] = []       # empty = no ladder
    # New context signals (alpha research additions)
    use_mtf_filter: bool = False      # +1 when 1h EMA(21) + 4h EMA(50) both agree with direction
    use_funding_filter: bool = False   # +1 when perpetual funding is extreme (fade the crowd)
    use_vwap_signal: bool = False      # +1 when price is extended from session VWAP
    vwap_dev_pct: float = 1.5         # % deviation from session VWAP to trigger signal
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


class DrawdownPoint(BaseModel):
    """Single point on the drawdown curve (always <= 0)."""
    time: int
    drawdown: int  # cents below running peak, always <= 0


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
    drawdown_series: List[DrawdownPoint] = []
    trade_log: List[TradeRecord]
    filter_stats: FilterStats
    # Risk-adjusted performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    # Streak & breakdown
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_win_cents: float = 0.0
    avg_loss_cents: float = 0.0
    win_loss_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    # Limit order stats (only populated when use_limit_order=True)
    limit_signals: int = 0           # total signals that attempted limit orders
    limit_fills: int = 0             # signals where the limit order was filled
    limit_fill_rate: float = 0.0     # fills / signals
    error: Optional[str] = None


# ── Optimizer ────────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    """Request body for the strategy optimizer endpoint."""
    start_ts: int
    end_ts: int
    mode: str = "btc_only"
    trials: int = 50                    # number of random configs to try (max 300)
    metric: str = "sharpe_ratio"        # field in BacktestResult to maximise
    top_n: int = 20                     # how many results to return (best N)
    # Fixed per-run params
    stake_cents: int = 100
    entry_price_cents: int = 50
    slippage_cents: int = 2
    min_trades: int = 5                 # discard configs with fewer trades
    # Search space toggles — set False to fix at default
    vary_min_score: bool = True
    vary_signals: bool = True
    vary_thresholds: bool = True
    vary_martingale: bool = False


class OptimizeTrialResult(BaseModel):
    """One row in the optimizer results table."""
    rank: int
    trial: int
    trades: int
    win_rate: float
    total_pnl_cents: int
    max_drawdown_cents: int
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    calmar_ratio: float
    expectancy: float
    # The exact BacktestParams that produced this result
    params: BacktestParams


class OptimizeResponse(BaseModel):
    """Returned by /api/optimize."""
    trials_run: int
    trials_valid: int          # trials with >= min_trades
    metric: str
    duration_seconds: float
    results: List[OptimizeTrialResult]


# ── Weather backtest models ───────────────────────────────────────────────────

class WeatherTradeRecord(BaseModel):
    time: str = ""
    ticker: str = ""
    title: str = ""
    series: str = ""
    entry_cents: int = 0
    result: str = ""        # "YES" or "NO" (market resolution)
    won: bool = False
    qty: int = 0
    pnl_cents: int = 0
    martingale_level: int = 0


class WeatherBacktestParams(BaseModel):
    start_ts: int
    end_ts: int
    series_tickers: List[str]   # e.g. ["KXHIGHNYD", "KXLOWNYD"]
    max_entry_cents: int = 20   # only enter when market price <= this
    min_entry_cents: int = 1    # floor — skip markets priced below this
    bet_side: str = "yes"       # "yes" | "no"
    stake_cents: int = 100      # dollars × 100 per trade
    use_martingale: bool = False
    martingale_multiplier: float = 2.0
    max_martingale_level: int = 5


class WeatherBacktestResult(BaseModel):
    total_markets_checked: int = 0
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_cents: int = 0
    avg_pnl_cents: float = 0.0
    avg_win_cents: float = 0.0
    avg_loss_cents: float = 0.0
    max_drawdown_cents: int = 0
    series_counts: dict = {}       # {series_ticker: markets_found}
    no_price_count: int = 0        # markets skipped because candlestick price unavailable
    equity_curve: List[EquityPoint] = []
    trade_log: List[WeatherTradeRecord] = []
