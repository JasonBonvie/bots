"""
Signal computation module for next 15-minute candle color prediction.

Strategy: predict whether the NEXT 15-minute BTC candle closes green or red.
Uses microstructure signals tuned for short-term candle-level prediction,
NOT trend-following indicators.

Scoring system (0-10 each direction with all strategies enabled):
  +1  Close position   — where price closed in the candle range (> 0.65 bull | < 0.35 bear)
  +1  Wick rejection   — long wick against direction signals reversal
  +1  Body strength    — strong full-body candle suggests continuation
  +1  RSI(5)           — short-period momentum (>55 bull | <45 bear)
  +1  Volume confirm   — high volume in candle direction adds conviction
  +1  Engulfing        — candle body fully swallows previous candle body (reversal signal)
  +1  Mean rev boost   — 4+ consecutive same-color candles → boost OPPOSITE direction
  +1  MTF Filter       — 1h EMA(21) + 4h EMA(50) both align with direction
  +1  VWAP Deviation   — price extended ≥1.5% from session VWAP (mean reversion)
  +1  Funding Rate     — extreme perpetual funding → crowd is overextended, fade the side

Veto / penalties:
  Doji candle: very small body → SKIP (no edge)
  Mean rev penalty: 4+ same-color candles → reduce SAME direction score by 1
"""

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone
from typing import Optional
import time
import logging
import urllib.request

from .models import SignalResult

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"

# Funding rate thresholds (absolute per-8h rate)
_FUNDING_BEAR_THRESHOLD = 0.0005   # longs stretched → mean reversion SHORT pressure
_FUNDING_BULL_THRESHOLD = -0.0002  # shorts stretched → mean reversion LONG pressure


def fetch_ohlcv_with_retry(
    n_bars: int = 50,
    aggregate: int = 15,
    interval: str = "",          # override string e.g. "1h", "4h" (takes priority over aggregate)
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> pd.DataFrame:
    """Fetch OHLCV data from Binance at the given minute aggregate or interval string."""

    last_error = None
    _interval = interval if interval else f"{aggregate}m"

    for attempt in range(max_retries):
        try:
            url = (
                f"{BINANCE_KLINES_URL}"
                f"?symbol=BTCUSDT&interval={_interval}&limit={n_bars + 1}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            if isinstance(data, dict) and data.get("code"):
                raise ValueError(f"Binance error: {data.get('msg', 'Unknown error')}")

            if not data:
                raise ValueError("Empty candle data returned")

            records = []
            for k in data:
                records.append({
                    "timestamp": pd.Timestamp(int(k[0]) // 1000, unit="s", tz="UTC"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            return df

        except Exception as e:
            last_error = e
            logger.warning(f"Binance fetch error ({_interval}), attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = min(retry_delay * (2 ** attempt), 5)
                time.sleep(wait_time)

    raise RuntimeError(f"Failed to fetch OHLCV ({_interval}) after {max_retries} retries: {last_error}")


def fetch_funding_rate_live() -> float:
    """Fetch the current perpetual funding rate from Binance Futures API."""
    try:
        url = f"{BINANCE_FUTURES_URL}?symbol=BTCUSDT"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        return float(data.get("lastFundingRate", 0.0))
    except Exception as e:
        logger.warning(f"Funding rate fetch failed: {e}")
        return 0.0


def compute_signals(
    symbol: str = "BTCUSDT",
    exchange: str = "Binance",
    bars: int = 100,
    rsi5_bull: float = 55.0,
    rsi5_bear: float = 45.0,
    close_pos_bull: float = 0.65,
    close_pos_bear: float = 0.35,
    body_ratio_min: float = 0.55,
    doji_threshold: float = 0.15,
    volume_ratio_min: float = 1.3,
    wick_ratio_min: float = 0.3,
    use_close_position: bool = True,
    use_wick_rejection: bool = True,
    use_body_strength: bool = True,
    use_rsi5: bool = True,
    use_volume_confirm: bool = True,
    consecutive_penalty: int = 4,
    min_score: float = 3.0,
    use_engulfing: bool = True,
    mean_reversion_boost: bool = True,
    # New context signals
    use_mtf_filter: bool = False,
    use_funding_filter: bool = False,
    use_vwap_signal: bool = False,
    vwap_dev_pct: float = 1.5,
) -> SignalResult:
    """
    Compute next-candle color prediction signals from BTC/USDT 15m data.
    All indicators are tuned for short-term (1-candle ahead) prediction.
    """

    df = fetch_ohlcv_with_retry(n_bars=bars, aggregate=15)
    logger.info(f"Successfully fetched {len(df)} candles from Binance")

    current_time = datetime.now(timezone.utc)

    # Use the last CLOSED candle (iloc[-2]) for signal computation.
    # iloc[-1] is the candle currently forming — not closed yet.
    if len(df) < 3:
        raise RuntimeError("Not enough candle data")

    # Last closed candle
    candle = df.iloc[-2]
    c_open  = float(candle["open"])
    c_high  = float(candle["high"])
    c_low   = float(candle["low"])
    c_close = float(candle["close"])
    c_vol   = float(candle["volume"])

    current_price = float(df["close"].iloc[-1])

    # --- Prev candle (one before the signal candle, for display + engulfing) ---
    prev_candle = df.iloc[-3]
    prev_open  = float(prev_candle["open"])
    prev_close = float(prev_candle["close"])
    if prev_close > prev_open:
        prev_candle_color = "GREEN"
    elif prev_close < prev_open:
        prev_candle_color = "RED"
    else:
        prev_candle_color = "NEUTRAL"

    price_change_pct = ((c_close - prev_close) / prev_close) * 100 if prev_close else 0.0

    # ── Derived candle metrics ────────────────────────────────────────
    candle_range = c_high - c_low
    body = abs(c_close - c_open)
    body_ratio = body / candle_range if candle_range > 0 else 0.0  # 0=doji, 1=full body

    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low
    upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0.0
    lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0.0

    # Close position: 1.0 = closed at high, 0.0 = closed at low
    close_position = (c_close - c_low) / candle_range if candle_range > 0 else 0.5

    # Is current candle green or red? (must be defined before engulfing)
    candle_is_green = c_close > c_open

    # ── Engulfing pattern (uses prev candle body vs signal candle body) ──
    prev_is_red   = prev_close < prev_open
    prev_is_green = prev_close > prev_open
    prev_body_top = max(prev_open, prev_close)
    prev_body_bot = min(prev_open, prev_close)
    sig_body_top = max(c_open, c_close)
    sig_body_bot = min(c_open, c_close)

    bullish_engulfing = (
        candle_is_green and prev_is_red and
        sig_body_bot <= prev_body_bot and sig_body_top >= prev_body_top
    )
    bearish_engulfing = (
        not candle_is_green and prev_is_green and
        sig_body_top >= prev_body_top and sig_body_bot <= prev_body_bot
    )

    # ── RSI(5) — short period for next-candle momentum ───────────────
    df["rsi5"] = ta.rsi(df["close"], length=5)
    rsi5 = float(df["rsi5"].iloc[-2]) if pd.notna(df["rsi5"].iloc[-2]) else 50.0

    # Keep RSI(14) for display only
    df["rsi"] = ta.rsi(df["close"], length=14)
    rsi = float(df["rsi"].iloc[-2]) if pd.notna(df["rsi"].iloc[-2]) else 50.0

    # ── EMA(8/21) — fast trend context for display ────────────────────
    df["ema8"]  = ta.ema(df["close"], length=8)
    df["ema21"] = ta.ema(df["close"], length=21)
    ema8  = float(df["ema8"].iloc[-2])  if pd.notna(df["ema8"].iloc[-2])  else current_price
    ema21 = float(df["ema21"].iloc[-2]) if pd.notna(df["ema21"].iloc[-2]) else current_price
    ema_signal = "BULL" if ema8 > ema21 else "BEAR"

    # ── MACD for display only ────────────────────────────────────────
    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    macd = float(macd_df["MACD_12_26_9"].iloc[-2]) if pd.notna(macd_df["MACD_12_26_9"].iloc[-2]) else 0.0
    macd_signal_line = float(macd_df["MACDs_12_26_9"].iloc[-2]) if pd.notna(macd_df["MACDs_12_26_9"].iloc[-2]) else 0.0

    # ── Volume ratio on closed candle ────────────────────────────────
    vol_sma = float(df["volume"].iloc[:-1].rolling(20).mean().iloc[-1])
    volume_ratio = c_vol / vol_sma if vol_sma > 0 else 1.0

    # ── Consecutive same-color candle count ──────────────────────────
    consecutive = 1
    colors = [(df["close"].iloc[i] > df["open"].iloc[i]) for i in range(-3, -1)]
    last_color = colors[-1]
    for color in reversed(colors[:-1]):
        if color == last_color:
            consecutive += 1
        else:
            break
    for i in range(-4, -min(10, len(df)) - 1, -1):
        try:
            is_green = df["close"].iloc[i] > df["open"].iloc[i]
            if is_green == last_color:
                consecutive += 1
            else:
                break
        except Exception:
            break

    # ── Session VWAP (reset at UTC midnight) ─────────────────────────
    vwap_price = 0.0
    vwap_dev = 0.0
    if use_vwap_signal or True:  # always compute for display
        try:
            signal_date = df.index[-2].date()
            session_mask = np.array([ts.date() == signal_date for ts in df.index[:-1]])
            session_df = df.iloc[:-1][session_mask]
            if len(session_df) > 0:
                cum_pv = (session_df["close"] * session_df["volume"]).cumsum()
                cum_vol = session_df["volume"].cumsum()
                vwap_price = float((cum_pv / cum_vol.replace(0, np.nan)).iloc[-1])
                vwap_dev = (c_close - vwap_price) / vwap_price * 100 if vwap_price > 0 else 0.0
        except Exception as e:
            logger.debug(f"VWAP computation failed: {e}")

    # ── Multi-Timeframe context (1h EMA21 + 4h EMA50) ────────────────
    htf_bias = "NEUTRAL"
    if use_mtf_filter:
        try:
            df_1h = fetch_ohlcv_with_retry(n_bars=50, interval="1h")
            df_4h = fetch_ohlcv_with_retry(n_bars=50, interval="4h")

            df_1h["ema21_1h"] = ta.ema(df_1h["close"], length=21)
            df_4h["ema50_4h"] = ta.ema(df_4h["close"], length=50)

            # Use last fully-closed 1h/4h candle (iloc[-2])
            last_1h_close = float(df_1h["close"].iloc[-2])
            ema21_1h = float(df_1h["ema21_1h"].iloc[-2]) if pd.notna(df_1h["ema21_1h"].iloc[-2]) else last_1h_close

            last_4h_close = float(df_4h["close"].iloc[-2])
            ema50_4h = float(df_4h["ema50_4h"].iloc[-2]) if pd.notna(df_4h["ema50_4h"].iloc[-2]) else last_4h_close

            one_h_bull = last_1h_close > ema21_1h
            four_h_bull = last_4h_close > ema50_4h

            if one_h_bull and four_h_bull:
                htf_bias = "BULL"
            elif not one_h_bull and not four_h_bull:
                htf_bias = "BEAR"
            else:
                htf_bias = "NEUTRAL"

            logger.info(f"MTF: 1h {'>' if one_h_bull else '<'} EMA21  4h {'>' if four_h_bull else '<'} EMA50 → {htf_bias}")
        except Exception as e:
            logger.warning(f"MTF fetch failed (skipping signal): {e}")
            htf_bias = "NEUTRAL"

    # ── Funding rate ─────────────────────────────────────────────────
    funding_rate = 0.0
    if use_funding_filter:
        funding_rate = fetch_funding_rate_live()
        logger.info(f"Funding rate: {funding_rate:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # SCORING — next-candle color prediction
    # ═══════════════════════════════════════════════════════════════

    score_up = 0.0
    score_dn = 0.0
    filters_passed = 0

    # ── Signal 1: Close Position ──────────────────────────────────
    if use_close_position:
        if close_position >= close_pos_bull:
            score_up += 1
            filters_passed += 1
        elif close_position <= close_pos_bear:
            score_dn += 1
            filters_passed += 1

    # ── Signal 2: Wick Rejection ──────────────────────────────────
    if use_wick_rejection:
        wick_reversal = wick_ratio_min + 0.1
        if candle_is_green and lower_wick_ratio >= wick_ratio_min and upper_wick_ratio < (wick_ratio_min - 0.1):
            score_up += 1
            filters_passed += 1
        elif not candle_is_green and upper_wick_ratio >= wick_ratio_min and lower_wick_ratio < (wick_ratio_min - 0.1):
            score_dn += 1
            filters_passed += 1
        elif candle_is_green and upper_wick_ratio >= wick_reversal:
            score_dn += 1
        elif not candle_is_green and lower_wick_ratio >= wick_reversal:
            score_up += 1

    # ── Signal 3: Body Strength ───────────────────────────────────
    if use_body_strength:
        if body_ratio >= body_ratio_min:
            if candle_is_green:
                score_up += 1
                filters_passed += 1
            else:
                score_dn += 1
                filters_passed += 1

    # ── Signal 4: RSI(5) Short Momentum ──────────────────────────
    if use_rsi5:
        if rsi5 > rsi5_bull:
            score_up += 1
            filters_passed += 1
        elif rsi5 < rsi5_bear:
            score_dn += 1
            filters_passed += 1

    # ── Signal 5: Volume Confirmation ────────────────────────────
    if use_volume_confirm:
        if volume_ratio >= volume_ratio_min:
            if candle_is_green:
                score_up += 1
                filters_passed += 1
            else:
                score_dn += 1
                filters_passed += 1

    # ── Signal 6: Engulfing Pattern ──────────────────────────────
    engulfing_signal = None
    if use_engulfing:
        if bullish_engulfing:
            score_up += 1
            filters_passed += 1
            engulfing_signal = "BULL"
        elif bearish_engulfing:
            score_dn += 1
            filters_passed += 1
            engulfing_signal = "BEAR"

    # ── Mean Reversion Penalty + Boost (Signal 7) ────────────────
    if consecutive >= consecutive_penalty:
        if candle_is_green:
            score_up = max(0, score_up - 1)
            if mean_reversion_boost:
                score_dn += 1
                filters_passed += 1
        else:
            score_dn = max(0, score_dn - 1)
            if mean_reversion_boost:
                score_up += 1
                filters_passed += 1

    # ── Signal 8: Multi-Timeframe Filter ─────────────────────────
    if use_mtf_filter and htf_bias != "NEUTRAL":
        if htf_bias == "BULL":
            score_up += 1
            filters_passed += 1
        elif htf_bias == "BEAR":
            score_dn += 1
            filters_passed += 1

    # ── Signal 9: VWAP Session Deviation ─────────────────────────
    if use_vwap_signal and vwap_price > 0:
        if vwap_dev <= -vwap_dev_pct:       # price extended below VWAP → mean reversion UP
            score_up += 1
            filters_passed += 1
        elif vwap_dev >= vwap_dev_pct:      # price extended above VWAP → mean reversion DOWN
            score_dn += 1
            filters_passed += 1

    # ── Signal 10: Funding Rate Extremes ─────────────────────────
    if use_funding_filter and funding_rate != 0.0:
        if funding_rate >= _FUNDING_BEAR_THRESHOLD:   # longs overextended → bear pressure
            score_dn += 1
            filters_passed += 1
        elif funding_rate <= _FUNDING_BULL_THRESHOLD:  # shorts overextended → bull pressure
            score_up += 1
            filters_passed += 1

    # ── Doji Veto ─────────────────────────────────────────────────
    doji = body_ratio < doji_threshold
    if doji:
        direction = "SKIP"
        confidence = "SKIP"
    else:
        if score_up >= min_score and score_up > score_dn:
            direction = "UP"
        elif score_dn >= min_score and score_dn > score_up:
            direction = "DOWN"
        else:
            direction = "SKIP"

        max_score = max(score_up, score_dn)
        if max_score >= 5:
            confidence = "HIGH"
        elif max_score >= min_score:
            confidence = "MEDIUM"
        else:
            confidence = "SKIP"

    # ── Filter labels for UI ──────────────────────────────────────
    rsi_filter      = "BULL" if rsi5 > rsi5_bull else ("BEAR" if rsi5 < rsi5_bear else "NEUTRAL")
    macd_filter     = "BULL" if macd > macd_signal_line else "BEAR"
    volume_filter   = "HIGH" if volume_ratio >= volume_ratio_min else "NORMAL"
    close_pos_filter = "BULL" if close_position >= close_pos_bull else ("BEAR" if close_position <= close_pos_bear else "NEUTRAL")

    logger.info(
        f"Next-candle signal: {direction} ({confidence}) | "
        f"close_pos={close_position:.2f} body={body_ratio:.2f} "
        f"upper_wick={upper_wick_ratio:.2f} lower_wick={lower_wick_ratio:.2f} "
        f"rsi5={rsi5:.1f} vol={volume_ratio:.2f}x consec={consecutive} "
        f"vwap_dev={vwap_dev:.2f}% htf={htf_bias} funding={funding_rate:.5f} "
        f"score↑{score_up} ↓{score_dn}"
    )

    filters_total = sum([
        use_close_position, use_wick_rejection, use_body_strength,
        use_rsi5, use_volume_confirm, use_engulfing, mean_reversion_boost,
        use_mtf_filter, use_vwap_signal, use_funding_filter,
    ])

    return SignalResult(
        timestamp=current_time,
        price=current_price,
        price_change_pct=round(price_change_pct, 2),
        rsi=round(rsi5, 2),
        ema8=round(ema8, 2),
        ema21=round(ema21, 2),
        ema_signal=ema_signal,
        macd=round(macd, 4),
        macd_signal_line=round(macd_signal_line, 4),
        volume_ratio=round(volume_ratio, 2),
        close_position=round(close_position, 2),
        score_up=round(score_up, 1),
        score_dn=round(score_dn, 1),
        direction=direction,
        confidence=confidence,
        filters_passed=filters_passed,
        filters_total=filters_total,
        rsi_filter=rsi_filter,
        macd_filter=macd_filter,
        volume_filter=volume_filter,
        close_pos_filter=close_pos_filter,
        prev_candle_color=prev_candle_color,
        mtf_bias=htf_bias,
        vwap=round(vwap_price, 2),
        vwap_dev_pct=round(vwap_dev, 2),
        funding_rate=round(funding_rate, 6),
    )


def get_individual_signal_states(result: SignalResult) -> dict:
    return {
        "rsi": {
            "value": result.rsi,
            "state": "bullish" if result.rsi > 55 else ("bearish" if result.rsi < 45 else "neutral"),
            "contribution": 1.0 if (result.rsi > 55 or result.rsi < 45) else 0.0,
            "label": "RSI(5)"
        },
        "ema": {
            "value": f"{result.ema8:.2f} / {result.ema21:.2f}",
            "state": "bullish" if result.ema_signal == "BULL" else "bearish",
            "contribution": 0.0,
            "label": "EMA(8/21)"
        },
        "macd": {
            "value": f"{result.macd:.4f}",
            "state": "bullish" if result.macd > result.macd_signal_line else "bearish",
            "contribution": 0.0,
            "label": "MACD"
        },
        "volume": {
            "value": f"{result.volume_ratio:.2f}x",
            "state": "active" if result.volume_ratio > 1.3 else "neutral",
            "contribution": 1.0 if result.volume_ratio > 1.3 else 0.0,
            "label": "Volume"
        },
        "close_position": {
            "value": f"{result.close_position:.2f}",
            "state": "bullish" if result.close_position > 0.65 else ("bearish" if result.close_position < 0.35 else "neutral"),
            "contribution": 1.0 if (result.close_position > 0.65 or result.close_position < 0.35) else 0.0,
            "label": "Close Position"
        },
        "mtf": {
            "value": result.mtf_bias,
            "state": "bullish" if result.mtf_bias == "BULL" else ("bearish" if result.mtf_bias == "BEAR" else "neutral"),
            "contribution": 1.0 if result.mtf_bias != "NEUTRAL" else 0.0,
            "label": "MTF(1h/4h)"
        },
        "vwap": {
            "value": f"{result.vwap_dev_pct:+.2f}%",
            "state": "bullish" if result.vwap_dev_pct <= -1.5 else ("bearish" if result.vwap_dev_pct >= 1.5 else "neutral"),
            "contribution": 1.0 if abs(result.vwap_dev_pct) >= 1.5 else 0.0,
            "label": "VWAP Dev"
        },
        "funding": {
            "value": f"{result.funding_rate:.5f}",
            "state": "bearish" if result.funding_rate >= 0.0005 else ("bullish" if result.funding_rate <= -0.0002 else "neutral"),
            "contribution": 1.0 if (result.funding_rate >= 0.0005 or result.funding_rate <= -0.0002) else 0.0,
            "label": "Funding"
        },
    }
