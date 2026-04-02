"""
Live price feed using Binance public API.
Polls real-time BTC/USDT price data — no auth required.
"""

import asyncio
import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import Set, Optional
from pydantic import BaseModel
import os

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=2"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"


class LivePrice(BaseModel):
    """Live price update model."""
    price: float
    timestamp: datetime
    candle_open: float
    candle_high: float
    candle_low: float
    candle_direction: str  # "bullish" or "bearish"
    change_from_open: float  # percentage change from candle open
    volume: float
    symbol: str
    exchange: str
    type: str = "live_price"


class BinanceLiveFeed:
    """
    Real-time BTC/USDT price feed via Binance public API.
    Polls every 2 seconds — no API key required.
    """

    def __init__(self, poll_interval: float = 2.0):
        self.symbol = "BTCUSDT"
        self.exchange = "Binance"
        self.poll_interval = float(os.getenv("PRICE_POLL_INTERVAL", str(poll_interval)))

        self.current_price: float = 0.0
        self.candle_open: float = 0.0
        self.candle_high: float = 0.0
        self.candle_low: float = 0.0
        self.candle_volume: float = 0.0
        self.candle_start_time: Optional[datetime] = None

        self.connected_clients: Set = set()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None

    def add_client(self, websocket):
        self.connected_clients.add(websocket)

    def remove_client(self, websocket):
        self.connected_clients.discard(websocket)

    def _get_candle_start_time(self) -> datetime:
        now = datetime.now(timezone.utc)
        minutes = (now.minute // 15) * 15
        return now.replace(minute=minutes, second=0, microsecond=0)

    def _should_reset_candle(self) -> bool:
        current_candle_start = self._get_candle_start_time()
        if self.candle_start_time is None:
            return True
        return current_candle_start > self.candle_start_time

    def _reset_candle(self, price: float, high: float, low: float, volume: float):
        self.candle_open = price
        self.candle_high = high
        self.candle_low = low
        self.candle_volume = volume
        self.candle_start_time = self._get_candle_start_time()
        logger.info(f"New candle started at {self.candle_start_time}, open: ${self.candle_open:,.2f}")

    def _update_candle(self, price: float, high: float, low: float, volume: float):
        if self._should_reset_candle():
            self._reset_candle(price, high, low, volume)
        elif self.candle_open == 0:
            self.candle_open = price
            self.candle_high = high
            self.candle_low = low

        self.current_price = price
        self.candle_high = max(self.candle_high, high)
        self.candle_low = min(self.candle_low, low) if self.candle_low > 0 else low
        self.candle_volume = volume

    def get_live_price(self) -> LivePrice:
        if self.candle_open > 0:
            change_pct = ((self.current_price - self.candle_open) / self.candle_open) * 100
            direction = "bullish" if self.current_price >= self.candle_open else "bearish"
        else:
            change_pct = 0.0
            direction = "bullish"

        return LivePrice(
            price=self.current_price,
            timestamp=datetime.now(timezone.utc),
            candle_open=self.candle_open,
            candle_high=self.candle_high,
            candle_low=self.candle_low,
            candle_direction=direction,
            change_from_open=round(change_pct, 4),
            volume=self.candle_volume,
            symbol=self.symbol,
            exchange=self.exchange
        )

    async def broadcast_price(self):
        if not self.connected_clients or self.current_price == 0:
            return

        live_price = self.get_live_price()
        message = live_price.model_dump_json()

        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.add(client)

        for client in disconnected:
            self.connected_clients.discard(client)

    async def _broadcast_loop(self):
        while self._running:
            try:
                await self.broadcast_price()
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(1)

    def _fetch_price_sync(self) -> Optional[tuple]:
        """Fetch latest 1-minute candle from Binance (blocking)."""
        try:
            req = urllib.request.Request(
                BINANCE_KLINES_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            if not isinstance(data, list) or not data:
                return None

            # Binance kline: [open_time, open, high, low, close, volume, ...]
            latest = data[-1]
            price = float(latest[4])   # close
            high = float(latest[2])
            low = float(latest[3])
            volume = float(latest[5])
            return price, high, low, volume

        except Exception as e:
            logger.debug(f"Binance fetch error: {e}")
            return None

    async def _poll_loop(self):
        logger.info("Starting Binance price polling (BTC/USDT)")

        poll_count = 0
        consecutive_errors = 0
        backoff = self.poll_interval

        while self._running:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._fetch_price_sync
                )

                if result:
                    price, high, low, volume = result
                    old_price = self.current_price
                    self._update_candle(price, high, low, volume)
                    consecutive_errors = 0
                    backoff = self.poll_interval
                    poll_count += 1

                    if old_price != price and old_price > 0:
                        direction = "🟢" if self.current_price >= self.candle_open else "🔴"
                        logger.debug(f"{direction} Binance BTC/USDT: ${self.current_price:,.2f}")

                    if poll_count % 30 == 1 and self.current_price > 0:
                        change = self.get_live_price().change_from_open
                        direction = "🟢" if self.current_price >= self.candle_open else "🔴"
                        logger.info(f"{direction} Binance BTC/USDT: ${self.current_price:,.2f} (candle: {change:+.3f}%)")
                else:
                    consecutive_errors += 1
                    backoff = min(30, self.poll_interval * (2 ** min(consecutive_errors, 4)))
                    if consecutive_errors % 5 == 0:
                        logger.warning(f"Binance: {consecutive_errors} consecutive errors, backing off {backoff:.0f}s")

                await asyncio.sleep(backoff)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll loop error: {e}")
                await asyncio.sleep(5)

        logger.info("Binance price polling stopped")

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("Binance live price feed started (BTC/USDT)")

    async def stop(self):
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None

        logger.info("Binance live price feed stopped")


# Global instance
live_price_feed = BinanceLiveFeed(
    poll_interval=float(os.getenv("PRICE_POLL_INTERVAL", "2.0"))
)
