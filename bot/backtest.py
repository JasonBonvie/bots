"""
Backtest engine for the BTC/Kalshi 15-minute momentum strategy.

All computation is synchronous and vectorized; the endpoint runs this via
asyncio.run_in_executor so the event loop stays unblocked.
"""
import json
import logging
import time
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from .models import (
    BacktestParams,
    BacktestResult,
    DrawdownPoint,
    LimitTier,
    EquityPoint,
    FilterStats,
    TradeRecord,
)

logger = logging.getLogger(__name__)

# Binance public klines endpoint (no API key required)
_BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
_BINANCE_PAGE_SIZE = 1000  # max candles Binance returns per call


class BacktestEngine:
    """
    Synchronous backtest engine — one instance per request, not shared.

    In btc_only mode the full pipeline runs inside run_btc_only().
    In kalshi mode the caller must pre-fetch Kalshi data asynchronously and
    pass it into run_kalshi_mode() because async I/O can't run inside an executor.
    """

    def __init__(self, params: BacktestParams):
        self.params = params

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_btc_ohlcv(self) -> pd.DataFrame:
        """
        Fetch 15-min BTC/USDT candles for [start_ts, end_ts] from Binance.
        Paginates forward using startTime until end_ts is covered.
        Returns a DataFrame indexed by UTC timestamp with open/high/low/close/volume.
        """
        p = self.params
        all_records: List[dict] = []
        # Fetch extra candles before start for indicator warmup (~25 x 15min = 375min)
        # RSI(5) needs ~5 bars, volume SMA needs 20 bars, consecutive count needs ~10
        warmup_seconds = 25 * 15 * 60
        start_ms = (p.start_ts - warmup_seconds) * 1000
        end_ms = p.end_ts * 1000

        while start_ms < end_ms:
            url = (
                f"{_BINANCE_KLINES}"
                f"?symbol=BTCUSDT&interval=15m"
                f"&startTime={start_ms}&endTime={end_ms}"
                f"&limit={_BINANCE_PAGE_SIZE}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = json.loads(resp.read().decode())
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError(f"Binance fetch failed: {exc}") from exc
                    time.sleep(1.0 * (attempt + 1))

            if data is None:
                raise ValueError("Binance: no response")
            if isinstance(data, dict) and data.get("code"):
                raise ValueError(f"Binance error: {data.get('msg', 'unknown')}")
            if not data:
                break

            for k in data:
                # Binance kline format: [open_time, open, high, low, close, volume, ...]
                ts_sec = int(k[0]) // 1000
                o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                vol = float(k[5])
                if o == 0 and c == 0:
                    continue
                all_records.append({
                    "timestamp": pd.Timestamp(ts_sec, unit="s", tz="UTC"),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": vol,
                })

            # Move start forward past the last candle we received
            last_open_time = int(data[-1][0])
            if last_open_time + 900_000 >= end_ms:
                break
            start_ms = last_open_time + 900_000  # next 15-min candle
            time.sleep(0.1)  # be polite to Binance rate limits

        if not all_records:
            raise ValueError("No OHLCV data returned for the requested date range.")

        df = pd.DataFrame(all_records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]

        # Don't trim to requested window here — warmup candles are needed
        # for indicator computation. Callers trim after computing indicators.

        if df.empty:
            raise ValueError("No OHLCV data within the requested date range after filtering.")

        return df

    # ------------------------------------------------------------------
    # Indicator computation (vectorized over full DataFrame)
    # ------------------------------------------------------------------

    def compute_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute next-candle color prediction signals — matches live signals.py exactly.
        Uses the last CLOSED candle (shift by 1) so we're predicting the NEXT candle.

        Signals (5 total, 0-5 score each direction):
          1. Close Position  — where price closed in the candle range
          2. Wick Rejection  — wick structure signals continuation or reversal
          3. Body Strength   — strong body = conviction, doji = skip
          4. RSI(5)          — short-period momentum
          5. Volume Confirm  — high volume in candle direction

        Mean reversion penalty: 4+ consecutive same-color candles reduce score by 1.
        Doji veto: body ratio < threshold → direction = SKIP.
        """
        p = self.params
        df = df.copy()

        # ── RSI(5) — short momentum ──────────────────────────────────
        df["rsi5"] = ta.rsi(df["close"], length=5)

        # Keep RSI(14) and EMA/MACD for display in filter stats
        df["rsi"] = ta.rsi(df["close"], length=14)
        df["ema8"] = ta.ema(df["close"], length=8)
        df["ema21"] = ta.ema(df["close"], length=21)
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd_df["MACD_12_26_9"]
        df["macd_sig"] = macd_df["MACDs_12_26_9"]

        # ── Volume ratio ─────────────────────────────────────────────
        vol_sma = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = (df["volume"] / vol_sma).fillna(1.0)

        # ── Candle metrics — use CLOSED candle (shift 1 for prediction) ──
        hi_lo = df["high"] - df["low"]
        df["close_position"] = ((df["close"] - df["low"]) / hi_lo.replace(0, np.nan)).fillna(0.5)
        df["body"] = (df["close"] - df["open"]).abs()
        df["body_ratio"] = (df["body"] / hi_lo.replace(0, np.nan)).fillna(0.0)
        df["candle_is_green"] = df["close"] > df["open"]
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["upper_wick_ratio"] = (df["upper_wick"] / hi_lo.replace(0, np.nan)).fillna(0.0)
        df["lower_wick_ratio"] = (df["lower_wick"] / hi_lo.replace(0, np.nan)).fillna(0.0)

        # Shift all candle metrics by 1 — we're computing signal from the CLOSED candle
        # to predict the NEXT candle (same as live signals.py using iloc[-2])
        for col in ["close_position", "body_ratio", "candle_is_green",
                    "upper_wick_ratio", "lower_wick_ratio", "volume_ratio", "rsi5",
                    "open", "close"]:
            df[f"s_{col}"] = df[col].shift(1)

        # Engulfing pattern — signal candle (shift 1) vs candle before it (shift 2)
        # Bullish: signal candle is green AND fully covers previous red body
        # Bearish: signal candle is red AND fully covers previous green body
        pp_open  = df["open"].shift(2)   # prev-prev candle open
        pp_close = df["close"].shift(2)  # prev-prev candle close
        pp_body_top = pd.concat([pp_open, pp_close], axis=1).max(axis=1)
        pp_body_bot = pd.concat([pp_open, pp_close], axis=1).min(axis=1)
        s_body_top  = pd.concat([df["s_open"], df["s_close"]], axis=1).max(axis=1)
        s_body_bot  = pd.concat([df["s_open"], df["s_close"]], axis=1).min(axis=1)

        pp_is_red   = pp_close < pp_open
        pp_is_green = pp_close > pp_open
        s_is_green  = df["s_close"] > df["s_open"]
        s_is_red    = df["s_close"] < df["s_open"]

        df["bull_engulfing"] = (
            s_is_green & pp_is_red &
            (s_body_bot <= pp_body_bot) &
            (s_body_top >= pp_body_top)
        ).fillna(False)
        df["bear_engulfing"] = (
            s_is_red & pp_is_green &
            (s_body_top >= pp_body_top) &
            (s_body_bot <= pp_body_bot)
        ).fillna(False)

        # ── Consecutive same-color candle count ──────────────────────
        # Vectorized: count consecutive same-color candles ending at each row
        green = df["candle_is_green"].astype(int)
        # Group by runs of same value, count within each group
        group = (green != green.shift()).cumsum()
        df["consecutive"] = green.groupby(group).cumcount() + 1
        df["s_consecutive"] = df["consecutive"].shift(1).fillna(1)

        # Previous candle color for display
        prev_close = df["close"].shift(1)
        prev_open = df["open"].shift(1)
        df["prev_candle_color"] = np.select(
            [prev_close > prev_open, prev_close < prev_open],
            ["GREEN", "RED"],
            default="NEUTRAL",
        )

        # ── Vectorized scoring ────────────────────────────────────────
        sc_up = pd.Series(0.0, index=df.index)
        sc_dn = pd.Series(0.0, index=df.index)

        # After shift(1) boolean columns become float64 (NaN forces it).
        # Cast to bool explicitly so ~ (bitwise NOT) works correctly.
        s_green = df["s_candle_is_green"].fillna(False).astype(bool)
        s_red = ~s_green

        # Signal 1: Close Position
        if p.use_close_position:
            sc_up += (df["s_close_position"] >= p.close_pos_bull).astype(float)
            sc_dn += (df["s_close_position"] <= p.close_pos_bear).astype(float)

        # Signal 2: Wick Rejection
        if p.use_wick_rejection:
            wr_min = p.wick_ratio_min
            wr_rev = wr_min + 0.1
            anti = max(0.0, wr_min - 0.1)
            green_clean = s_green & (df["s_lower_wick_ratio"] >= wr_min) & (df["s_upper_wick_ratio"] < anti)
            red_clean   = s_red   & (df["s_upper_wick_ratio"] >= wr_min) & (df["s_lower_wick_ratio"] < anti)
            green_rev   = s_green & (df["s_upper_wick_ratio"] >= wr_rev)
            red_rev     = s_red   & (df["s_lower_wick_ratio"] >= wr_rev)
            sc_up += green_clean.astype(float) + red_rev.astype(float)
            sc_dn += red_clean.astype(float) + green_rev.astype(float)

        # Signal 3: Body Strength
        if p.use_body_strength:
            strong_green = s_green & (df["s_body_ratio"] >= p.body_ratio_min)
            strong_red   = s_red   & (df["s_body_ratio"] >= p.body_ratio_min)
            sc_up += strong_green.astype(float)
            sc_dn += strong_red.astype(float)

        # Signal 4: RSI(5)
        if p.use_rsi5:
            sc_up += (df["s_rsi5"] > p.rsi5_bull).astype(float)
            sc_dn += (df["s_rsi5"] < p.rsi5_bear).astype(float)

        # Signal 5: Volume Confirmation
        if p.use_volume_confirm:
            high_vol_green = (df["s_volume_ratio"] >= p.volume_ratio_min) & s_green
            high_vol_red   = (df["s_volume_ratio"] >= p.volume_ratio_min) & s_red
            sc_up += high_vol_green.astype(float)
            sc_dn += high_vol_red.astype(float)

        # Signal 6: Engulfing Pattern
        if p.use_engulfing:
            sc_up += df["bull_engulfing"].astype(float)
            sc_dn += df["bear_engulfing"].astype(float)

        # Mean Reversion Penalty + optional Boost
        penalty_green = (df["s_consecutive"] >= p.consecutive_penalty) & s_green
        penalty_red   = (df["s_consecutive"] >= p.consecutive_penalty) & s_red
        sc_up = (sc_up - penalty_green.astype(float)).clip(lower=0)
        sc_dn = (sc_dn - penalty_red.astype(float)).clip(lower=0)
        if p.mean_reversion_boost:
            sc_dn += penalty_green.astype(float)  # boost bear when green streak ends
            sc_up += penalty_red.astype(float)    # boost bull when red streak ends

        df["score_up"] = sc_up
        df["score_dn"] = sc_dn

        # Doji veto + direction
        doji = df["s_body_ratio"] < p.doji_threshold
        df["direction"] = np.select(
            [
                doji,
                (sc_up >= p.min_score) & (sc_up > sc_dn),
                (sc_dn >= p.min_score) & (sc_dn > sc_up),
            ],
            ["SKIP", "UP", "DOWN"],
            default="SKIP",
        )

        # Filter labels for FilterStats
        df["_ema_label"] = np.where(df["ema8"] > df["ema21"], "bull", "bear")
        df["_rsi_label"] = np.select(
            [(df["rsi"] > 50) & (df["rsi"] < 70), (df["rsi"] > 30) & (df["rsi"] < 50)],
            ["bull", "bear"], default="neutral"
        )
        df["_macd_label"] = np.where(df["macd"] > df["macd_sig"], "bull", "bear")
        df["_cp_label"] = np.select(
            [df["s_close_position"] >= p.close_pos_bull, df["s_close_position"] <= p.close_pos_bear],
            ["bull", "bear"], default="neutral"
        )

        return df

    # ------------------------------------------------------------------
    # Martingale stake helper
    # ------------------------------------------------------------------

    def _martingale_stake(self, base_cents: int, entry_cents: int, level: int) -> int:
        """
        Return number of contracts for this trade using the same dollar-doubling
        logic as the live bot: multiply BASE DOLLAR STAKE by multiplier^level,
        then convert to contracts via entry price.
        """
        p = self.params
        dollar_stake = int(base_cents * (p.martingale_multiplier ** level))
        return max(1, dollar_stake // max(1, entry_cents))

    # ------------------------------------------------------------------
    # BTC-only mode
    # ------------------------------------------------------------------

    def run_btc_only(self, df_btc: "pd.DataFrame | None" = None) -> BacktestResult:
        """
        Fast mode: signal direction vs next-candle direction determines outcome.
        Entry price is params.entry_price_cents (assumed flat).

        df_btc: optional pre-fetched raw OHLCV dataframe — pass in when running
                many trials so the data is fetched only once.
        """
        t0 = time.time()
        p = self.params

        df = df_btc.copy() if df_btc is not None else self.fetch_btc_ohlcv()
        df = self.compute_indicators_vectorized(df)
        df = df.dropna(subset=["s_rsi5", "s_close_position", "s_body_ratio"])

        # Trim to requested window (warmup candles were kept for indicator computation)
        start_ts = pd.Timestamp(p.start_ts, unit="s", tz="UTC")
        end_ts = pd.Timestamp(p.end_ts, unit="s", tz="UTC")
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

        trade_log: List[TradeRecord] = []
        equity = 0
        equity_curve: List[EquityPoint] = []
        peak_equity = 0
        max_drawdown = 0
        martingale_level = 0  # tracks consecutive losses for martingale

        rows = df.reset_index()
        n = len(rows)

        for i in range(n - 1):  # stop before last row (need next candle)
            row = rows.iloc[i]
            direction = row["direction"]
            ts_int = int(row["timestamp"].timestamp())

            if direction == "SKIP":
                equity_curve.append(EquityPoint(time=ts_int, equity=equity))
                continue

            # Candle confirmation filter
            if p.require_prev_candle_match:
                prev_color = row.get("prev_candle_color", "NEUTRAL")
                if direction == "UP" and prev_color != "GREEN":
                    equity_curve.append(EquityPoint(time=ts_int, equity=equity))
                    continue
                if direction == "DOWN" and prev_color != "RED":
                    equity_curve.append(EquityPoint(time=ts_int, equity=equity))
                    continue

            next_close = float(rows.iloc[i + 1]["close"])
            current_close = float(row["close"])
            won = (next_close > current_close) if direction == "UP" else (next_close < current_close)

            entry = min(99, p.entry_price_cents + p.slippage_cents)
            exit_price = 100 if won else 0

            # Martingale: double dollar stake each loss, same logic as live bot
            if p.use_martingale and martingale_level > 0:
                qty = self._martingale_stake(p.stake_cents, entry, martingale_level)
            else:
                qty = max(1, p.stake_cents // entry)

            pnl = (exit_price - entry) * qty

            # Update martingale level for next trade
            if p.use_martingale:
                if won:
                    martingale_level = 0
                else:
                    martingale_level = min(martingale_level + 1, p.max_martingale_level)

            equity += pnl
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, peak_equity - equity)

            trade_log.append(TradeRecord(
                time=ts_int,
                direction=direction,
                score_up=float(row["score_up"]),
                score_dn=float(row["score_dn"]),
                btc_price=current_close,
                entry_price_cents=entry,
                exit_price_cents=exit_price,
                result="WIN" if won else "LOSS",
                pnl_cents=pnl,
            ))
            equity_curve.append(EquityPoint(time=ts_int, equity=equity))

        return self._build_result(df, trade_log, equity_curve, max_drawdown, time.time() - t0)

    # ------------------------------------------------------------------
    # Kalshi mode
    # ------------------------------------------------------------------

    def run_kalshi_mode(
        self,
        df_btc: pd.DataFrame,
        kalshi_markets: List[Dict],
        kalshi_candles: Dict[str, List[Dict]],
    ) -> BacktestResult:
        """
        Full mode: uses real Kalshi settlement outcomes and real entry prices.

        kalshi_markets: list of settled market dicts, each with keys:
            ticker, close_time (ISO string), result ("yes"/"no"), floor_strike
        kalshi_candles: ticker -> list of 1-min candlestick dicts, each with:
            end_period_ts, yes_ask {open, close, ...} (dollar strings)
        """
        t0 = time.time()
        p = self.params

        df = self.compute_indicators_vectorized(df_btc)
        df = df.dropna(subset=["macd", "rsi", "ema8", "ema21"])

        # Trim to requested window (warmup candles were kept for indicator computation)
        start_ts = pd.Timestamp(p.start_ts, unit="s", tz="UTC")
        end_ts = pd.Timestamp(p.end_ts, unit="s", tz="UTC")
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

        # Build lookup: market close boundary ts -> market dict.
        # Each KXBTC15M market opens at candle T close and resolves at candle T+900.
        # We index by close_time (snapped to 15-min boundary) so a signal at T
        # can look up the market resolving at T+900.
        market_by_close: Dict[int, Dict] = {}
        for m in kalshi_markets:
            try:
                close_ts = int(pd.Timestamp(m["close_time"]).timestamp())
                close_ts = (close_ts // 900) * 900  # snap to 15-min grid
                market_by_close[close_ts] = m
            except Exception:
                continue

        trade_log: List[TradeRecord] = []
        equity = 0
        equity_curve: List[EquityPoint] = []
        peak_equity = 0
        max_drawdown = 0
        limit_signals = 0
        limit_fills = 0
        martingale_level = 0  # tracks consecutive losses

        rows = df.reset_index()
        qty_override: Optional[int] = None  # set by ladder simulation, consumed once
        for _, row in rows.iterrows():
            qty_override = None  # reset each candle
            direction = row["direction"]
            ts_int = int(row["timestamp"].timestamp())

            # Signal fires at candle T close → trade the market that resolves at T+900
            next_candle_ts = ts_int + 900

            if direction == "SKIP" or next_candle_ts not in market_by_close:
                equity_curve.append(EquityPoint(time=ts_int, equity=equity))
                continue

            # Candle confirmation filter
            if p.require_prev_candle_match:
                prev_color = row.get("prev_candle_color", "NEUTRAL")
                if direction == "UP" and prev_color != "GREEN":
                    equity_curve.append(EquityPoint(time=ts_int, equity=equity))
                    continue
                if direction == "DOWN" and prev_color != "RED":
                    equity_curve.append(EquityPoint(time=ts_int, equity=equity))
                    continue

            market = market_by_close[next_candle_ts]
            ticker = market.get("ticker", "")
            market_result = market.get("result", "")  # "yes" or "no"

            if direction == "UP":
                won = market_result == "yes"
            else:
                won = market_result == "no"

            candles = kalshi_candles.get(ticker, [])

            if p.use_limit_order:
                # --- Limit order ladder simulation ---
                limit_signals += 1

                def _check_tier_fill(price_cents: int, window_minutes: int) -> bool:
                    """Returns True if the ask price dipped to price_cents within window_minutes candles."""
                    for c in candles[:window_minutes]:
                        try:
                            ask_low = c.get("yes_ask", {})
                            raw_low = ask_low.get("low_dollars") or ask_low.get("low")
                            if raw_low is not None:
                                if int(float(raw_low) * 100) <= price_cents:
                                    return True
                        except (TypeError, ValueError):
                            continue
                    return False

                # Check primary tier
                fills: List[tuple] = []  # (price_cents, qty)
                if _check_tier_fill(p.limit_price_cents, p.limit_window_minutes):
                    primary_qty = max(1, p.stake_cents // max(1, p.limit_price_cents))
                    fills.append((p.limit_price_cents, primary_qty))

                # Check each ladder tier
                for tier in p.limit_ladder:
                    if _check_tier_fill(tier.price_cents, tier.window_minutes):
                        tier_qty = max(1, int(tier.stake_dollars * 100) // max(1, tier.price_cents))
                        fills.append((tier.price_cents, tier_qty))

                if not fills:
                    equity_curve.append(EquityPoint(time=ts_int, equity=equity))
                    continue

                limit_fills += 1
                # Weighted average entry across all filled tiers
                total_qty = sum(q for _, q in fills)
                total_cost = sum(pr * q for pr, q in fills)
                entry = total_cost // total_qty
                qty_override = total_qty  # use combined qty instead of default calculation
            else:
                # --- Market order: use first candle's ask open ---
                entry = p.entry_price_cents
                if candles:
                    try:
                        ask_data = candles[0].get("yes_ask", {})
                        raw_ask = ask_data.get("open_dollars") or ask_data.get("open")
                        if raw_ask is not None:
                            entry = max(1, min(99, int(float(raw_ask) * 100)))
                    except (TypeError, ValueError):
                        pass
                entry = min(99, entry + p.slippage_cents)

            exit_price = 100 if won else 0

            # Martingale: double dollar stake each loss, reset on win
            # qty_override is set when a ladder fills (combined contracts across tiers)
            if qty_override is not None:
                qty = qty_override
            elif p.use_martingale and martingale_level > 0:
                qty = self._martingale_stake(p.stake_cents, entry, martingale_level)
            else:
                qty = max(1, p.stake_cents // entry)

            pnl = (exit_price - entry) * qty

            # Update martingale level for next trade
            if p.use_martingale:
                if won:
                    martingale_level = 0
                else:
                    martingale_level = min(martingale_level + 1, p.max_martingale_level)

            equity += pnl
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, peak_equity - equity)

            trade_log.append(TradeRecord(
                time=ts_int,
                direction=direction,
                score_up=float(row["score_up"]),
                score_dn=float(row["score_dn"]),
                btc_price=float(row["close"]),
                entry_price_cents=entry,
                exit_price_cents=exit_price,
                result="WIN" if won else "LOSS",
                pnl_cents=pnl,
                market_ticker=ticker,
            ))
            equity_curve.append(EquityPoint(time=ts_int, equity=equity))

        result = self._build_result(df, trade_log, equity_curve, max_drawdown, time.time() - t0)
        result.limit_signals = limit_signals
        result.limit_fills = limit_fills
        result.limit_fill_rate = round(limit_fills / limit_signals, 4) if limit_signals > 0 else 0.0
        return result

    # ------------------------------------------------------------------
    # Shared result builder
    # ------------------------------------------------------------------

    def _build_result(
        self,
        df: pd.DataFrame,
        trade_log: List[TradeRecord],
        equity_curve: List[EquityPoint],
        max_drawdown: int,
        duration: float,
    ) -> BacktestResult:
        wins   = [t for t in trade_log if t.result == "WIN"]
        losses = [t for t in trade_log if t.result == "LOSS"]
        trades_taken = len(trade_log)
        total_pnl = sum(t.pnl_cents for t in trade_log)
        win_rate  = len(wins) / trades_taken if trades_taken else 0.0
        avg_pnl   = total_pnl / trades_taken if trades_taken else 0.0

        stake = self.params.stake_cents
        expectancy = avg_pnl / stake if stake > 0 else 0.0

        # ── Per-trade dollar returns (for Sharpe / Sortino) ───────────
        trade_returns = np.array([
            t.pnl_cents / (t.entry_price_cents * max(1, stake // max(1, t.entry_price_cents)))
            for t in trade_log
        ]) if trade_log else np.array([])

        # ── Sharpe Ratio (annualised) ─────────────────────────────────
        # Assume 96 trades/day possible (96 × 15-min candles per 24h).
        # Actual trade frequency is win_rate × signals; annualise by scaling.
        candles_per_year = 96 * 365
        if len(trade_returns) >= 2:
            mu  = float(np.mean(trade_returns))
            std = float(np.std(trade_returns, ddof=1))
            # trades_per_year estimate: extrapolate from backtest window
            days = max(1, (self.params.end_ts - self.params.start_ts) / 86400)
            tpy  = (trades_taken / days) * 365
            ann  = float(np.sqrt(tpy))
            sharpe  = round((mu / std) * ann, 3) if std > 0 else 0.0
            neg     = trade_returns[trade_returns < 0]
            downstd = float(np.std(neg, ddof=1)) if len(neg) >= 2 else std
            sortino = round((mu / downstd) * ann, 3) if downstd > 0 else 0.0
        else:
            sharpe = sortino = 0.0

        # ── Profit Factor ─────────────────────────────────────────────
        gross_profit = sum(t.pnl_cents for t in wins)
        gross_loss   = abs(sum(t.pnl_cents for t in losses))
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        # ── Win / Loss averages & ratio ───────────────────────────────
        avg_win  = round(gross_profit / len(wins),   2) if wins   else 0.0
        avg_loss = round(gross_loss   / len(losses), 2) if losses else 0.0
        wl_ratio = round(avg_win / avg_loss, 3) if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0

        # ── Streaks ───────────────────────────────────────────────────
        max_cw = max_cl = cur_cw = cur_cl = 0
        for t in trade_log:
            if t.result == "WIN":
                cur_cw += 1; cur_cl = 0
                max_cw = max(max_cw, cur_cw)
            else:
                cur_cl += 1; cur_cw = 0
                max_cl = max(max_cl, cur_cl)

        # ── Drawdown series ───────────────────────────────────────────
        peak = 0
        drawdown_series: List[DrawdownPoint] = []
        for ep in equity_curve:
            peak = max(peak, ep.equity)
            dd   = ep.equity - peak          # always <= 0
            drawdown_series.append(DrawdownPoint(time=ep.time, drawdown=dd))

        # ── Max drawdown % ────────────────────────────────────────────
        max_dd_pct = round((max_drawdown / peak) * 100, 2) if peak > 0 else 0.0

        # ── Calmar Ratio (annualised return / max drawdown) ───────────
        if max_drawdown > 0 and trades_taken > 0:
            days = max(1, (self.params.end_ts - self.params.start_ts) / 86400)
            ann_return = (total_pnl / max(1, stake)) * (365 / days)
            calmar = round(ann_return / (max_drawdown / max(1, stake)), 3)
        else:
            calmar = 0.0

        # ── Recovery Factor ───────────────────────────────────────────
        recovery = round(total_pnl / max_drawdown, 3) if max_drawdown > 0 else 0.0

        # ── FilterStats ───────────────────────────────────────────────
        valid = df.dropna(subset=["_ema_label", "_rsi_label", "_macd_label", "_cp_label"])
        fs = FilterStats(
            total_candles=len(valid),
            ema_bull=int((valid["_ema_label"] == "bull").sum()),
            ema_bear=int((valid["_ema_label"] == "bear").sum()),
            rsi_bull=int((valid["_rsi_label"] == "bull").sum()),
            rsi_bear=int((valid["_rsi_label"] == "bear").sum()),
            rsi_neutral=int((valid["_rsi_label"] == "neutral").sum()),
            macd_bull=int((valid["_macd_label"] == "bull").sum()),
            macd_bear=int((valid["_macd_label"] == "bear").sum()),
            close_pos_bull=int((valid["_cp_label"] == "bull").sum()),
            close_pos_bear=int((valid["_cp_label"] == "bear").sum()),
            close_pos_neutral=int((valid["_cp_label"] == "neutral").sum()),
            signals_up=int((valid["direction"] == "UP").sum()),
            signals_down=int((valid["direction"] == "DOWN").sum()),
            signals_skip=int((valid["direction"] == "SKIP").sum()),
        )

        return BacktestResult(
            params=self.params,
            duration_seconds=round(duration, 2),
            total_candles=len(valid),
            total_signals=fs.signals_up + fs.signals_down,
            trades_taken=trades_taken,
            wins=len(wins),
            losses=len(losses),
            win_rate=round(win_rate, 4),
            total_pnl_cents=total_pnl,
            avg_pnl_cents=round(avg_pnl, 2),
            expectancy=round(expectancy, 4),
            max_drawdown_cents=max_drawdown,
            max_drawdown_pct=max_dd_pct,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            trade_log=trade_log,
            filter_stats=fs,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor if profit_factor != float("inf") else 999.0,
            calmar_ratio=calmar,
            recovery_factor=recovery,
            max_consecutive_wins=max_cw,
            max_consecutive_losses=max_cl,
            avg_win_cents=avg_win,
            avg_loss_cents=avg_loss,
            win_loss_ratio=wl_ratio if wl_ratio != float("inf") else 999.0,
        )
