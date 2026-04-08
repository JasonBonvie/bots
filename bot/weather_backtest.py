"""
Weather Prediction Market Backtester

Strategy: consistently bet on low-probability events across daily weather markets.
For each settled market in the date range whose pre-settlement price is at or below
max_entry_cents, simulate a buy on the configured side and track outcome at settlement.

Price source: Kalshi's `previous_yes_ask_dollars` / `previous_yes_bid_dollars` fields
on the settled market object (candlestick history is not available for weather markets).
  - YES side entry: previous_yes_ask_dollars × 100
  - NO  side entry: 100 − (previous_yes_bid_dollars × 100)   [NO ask ≈ 100 − YES bid]
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

        # Track how many markets were fetched per series
        series_counts: Dict[str, int] = {}
        no_price_count = 0
        price_filtered_count = 0
        for m in all_markets:
            s = m.get("_series", "unknown")
            series_counts[s] = series_counts.get(s, 0) + 1

        for market in all_markets:
            ticker = market.get("ticker", "")
            result = market.get("result", "")   # "yes" or "no"
            close_time = market.get("close_time", "")
            title = market.get("title", ticker)
            series = market.get("_series", "")

            if not ticker or result not in ("yes", "no"):
                continue

            # Get entry price from previous_yes_ask / previous_yes_bid fields.
            # These are the last-known prices before settlement — the best proxy
            # available since Kalshi does not retain candlestick history for
            # settled weather markets.
            entry_price = self._extract_entry_price(market, p.bet_side)
            if entry_price is None:
                no_price_count += 1
                logger.debug(f"No price available for {ticker} — skipping")
                continue

            # Filter by probability window
            if entry_price > p.max_entry_cents or entry_price < p.min_entry_cents:
                price_filtered_count += 1
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

            close_ts = _parse_ts(close_time)
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
            series_counts=series_counts,
            no_price_count=no_price_count,
            price_filtered_count=price_filtered_count,
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

    def _extract_entry_price(self, market: Dict, side: str) -> Optional[int]:
        """
        Extract the best-available entry price from the settled market object.

        Uses `previous_yes_ask_dollars` / `previous_yes_bid_dollars` — the last
        known pre-settlement prices that Kalshi includes in the settled market dict.

        YES side: entry = previous_yes_ask × 100
        NO  side: entry = 100 − (previous_yes_bid × 100)
                  (NO ask ≈ 100 − YES bid in a binary market)

        Falls back to `previous_price_dollars` if ask/bid fields are missing.
        Returns None only if no price field is present at all.
        """
        def _to_cents(field: str) -> Optional[int]:
            v = market.get(field)
            if v is None:
                return None
            try:
                f = float(v)
                # Kalshi returns dollar strings like "0.0100"; multiply by 100
                cents = int(round(f * 100)) if f <= 1.0 else int(round(f))
                return max(1, min(99, cents))
            except (ValueError, TypeError):
                return None

        if side == "yes":
            price = _to_cents("previous_yes_ask_dollars")
            if price is None:
                price = _to_cents("previous_price_dollars")
            return price
        else:
            # NO ask ≈ 100 − YES bid
            yes_bid = _to_cents("previous_yes_bid_dollars")
            if yes_bid is not None:
                no_ask = 100 - yes_bid
                return max(1, min(99, no_ask))
            # Fallback: infer from previous_price (treat as YES mid, flip for NO)
            yes_mid = _to_cents("previous_price_dollars")
            if yes_mid is not None:
                return max(1, min(99, 100 - yes_mid))
            return None


def _parse_ts(time_str: str) -> int:
    if not time_str:
        return 0
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return 0
