"""
Weather Prediction Market Backtester

Strategy: consistently bet on low-probability events across daily weather markets.
For each settled market in the date range whose opening price is at or below
max_entry_cents, simulate a buy on the configured side and track outcome at settlement.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict

from bot.models import (
    WeatherBacktestParams,
    WeatherBacktestResult,
    WeatherTradeRecord,
    EquityPoint,
)

logger = logging.getLogger(__name__)


class WeatherBacktestEngine:

    def __init__(self, kalshi_client=None):
        self._client = kalshi_client

    async def run(self, params: WeatherBacktestParams) -> WeatherBacktestResult:
        if not self._client:
            raise ValueError("Kalshi client not configured — set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")

        p = params
        trades: List[WeatherTradeRecord] = []
        equity = 0
        equity_curve: List[EquityPoint] = []
        martingale_level = 0
        total_markets_checked = 0

        # Collect + sort all settled markets across requested series
        all_markets: List[Dict] = []
        for series in p.series_tickers:
            markets = await self._client.get_settled_markets(
                series_ticker=series,
                min_close_ts=p.start_ts,
                max_close_ts=p.end_ts,
                limit=200,
            )
            for m in markets:
                m["_series"] = series
            all_markets.extend(markets)
            logger.info(f"Fetched {len(markets)} settled markets for {series}")

        all_markets.sort(key=lambda m: m.get("close_time", ""))
        total_markets_checked = len(all_markets)

        for market in all_markets:
            ticker = market.get("ticker", "")
            result = market.get("result", "")   # "yes" or "no"
            close_time = market.get("close_time", "")
            title = market.get("title", ticker)
            series = market.get("_series", "")

            if not ticker or result not in ("yes", "no"):
                continue

            # Get entry price from market open (first hourly candle)
            close_ts = _parse_ts(close_time)
            market_open_ts = close_ts - 86400  # look back up to 24h
            entry_price = await self._get_open_price(ticker, p.bet_side, market_open_ts, close_ts)
            if entry_price is None:
                continue

            # Filter by probability window
            if entry_price > p.max_entry_cents:
                continue
            if entry_price < p.min_entry_cents:
                continue

            won = (result == p.bet_side)
            exit_price = 100 if won else 0

            # Stake sizing with optional martingale
            if p.use_martingale and martingale_level > 0:
                multiplier = p.martingale_multiplier ** martingale_level
                stake = int(p.stake_cents * multiplier)
            else:
                stake = p.stake_cents

            qty = max(1, stake // max(1, entry_price))
            pnl = (exit_price - entry_price) * qty
            equity += pnl

            equity_curve.append(EquityPoint(time=close_ts, equity=equity))

            trades.append(WeatherTradeRecord(
                time=close_time,
                ticker=ticker,
                title=title,
                series=series,
                entry_cents=entry_price,
                result=result.upper(),
                won=won,
                qty=qty,
                pnl_cents=pnl,
                martingale_level=martingale_level,
            ))

            # Update martingale level
            if p.use_martingale:
                if won:
                    martingale_level = 0
                elif martingale_level < p.max_martingale_level:
                    martingale_level += 1

        wins = sum(1 for t in trades if t.won)
        losses = len(trades) - wins
        total_pnl = sum(t.pnl_cents for t in trades)
        win_rate = wins / len(trades) if trades else 0.0
        avg_pnl = total_pnl / len(trades) if trades else 0.0
        avg_win = sum(t.pnl_cents for t in trades if t.won) / max(1, wins)
        avg_loss = sum(t.pnl_cents for t in trades if not t.won) / max(1, losses) if losses else 0.0

        # Max drawdown
        peak = 0
        max_dd = 0
        running = 0
        for t in trades:
            running += t.pnl_cents
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        return WeatherBacktestResult(
            total_markets_checked=total_markets_checked,
            trades_taken=len(trades),
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 4),
            total_pnl_cents=total_pnl,
            avg_pnl_cents=round(avg_pnl, 1),
            avg_win_cents=round(avg_win, 1),
            avg_loss_cents=round(avg_loss, 1),
            max_drawdown_cents=max_dd,
            equity_curve=equity_curve,
            trade_log=trades,
        )

    async def _get_open_price(
        self,
        ticker: str,
        side: str,
        start_ts: int,
        end_ts: int,
    ) -> Optional[int]:
        """Get the opening price for the configured side from the first hourly candle."""
        try:
            candles = await self._client.get_historical_candlesticks(
                ticker, start_ts=start_ts, end_ts=end_ts, period_interval=60
            )
            if not candles:
                return None

            first = candles[0]
            if side == "yes":
                ask = first.get("yes_ask", {}) or {}
            else:
                ask = first.get("no_ask", {}) or {}

            raw = ask.get("open_dollars") or ask.get("open")

            # fallback: use mid-price from yes_bid + yes_ask
            if raw is None and side == "yes":
                bid = first.get("yes_bid", {}) or {}
                raw = bid.get("open_dollars") or bid.get("open")

            if raw is None:
                return None

            price = int(float(raw) * 100)
            return max(1, min(99, price))

        except Exception as exc:
            logger.debug(f"Could not get price for {ticker}: {exc}")
            return None


def _parse_ts(time_str: str) -> int:
    if not time_str:
        return 0
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return 0
