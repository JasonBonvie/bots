"""
Scheduler for 15-minute candle-close synchronization.

This module handles timing the signal computation to align with
Binance 15-minute candle closes (:00, :15, :30, :45 of each hour).
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable, List, Set
import logging

from fastapi import WebSocket

from bot.signals import compute_signals
from bot.models import SignalResult
from bot.bot_manager import bot_manager

logger = logging.getLogger(__name__)


class SignalScheduler:
    """Scheduler that syncs signal computation to 15-minute candle closes."""

    def __init__(self):
        self.connected_clients: Set[WebSocket] = set()
        self.latest_signal: SignalResult | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._auto_trade_enabled: bool = True  # Enable/disable auto-trading
        self._kalshi_feed = None  # Set via set_kalshi_feed()

    def set_kalshi_feed(self, kalshi_feed):
        """Set the Kalshi feed for auto-trading."""
        self._kalshi_feed = kalshi_feed
        logger.info("Kalshi feed connected to scheduler for auto-trading")

    def set_auto_trade(self, enabled: bool):
        """Enable or disable auto-trading."""
        self._auto_trade_enabled = enabled
        logger.info(f"Auto-trading {'enabled' if enabled else 'disabled'}")

    def add_client(self, websocket: WebSocket):
        """Add a WebSocket client to receive broadcasts."""
        self.connected_clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")

    def remove_client(self, websocket: WebSocket):
        """Remove a WebSocket client from broadcasts."""
        self.connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")

    @staticmethod
    def seconds_until_next_15m() -> float:
        """Calculate seconds until the next 15-minute boundary.

        Returns seconds until the next :00, :15, :30, or :45 minute mark.
        Adds a small buffer (2 seconds) to ensure the candle has fully closed.
        """
        now = datetime.utcnow()

        # Find the next 15-minute boundary
        current_minute = now.minute
        current_second = now.second
        current_microsecond = now.microsecond

        # Calculate minutes until next 15-minute mark
        minutes_past_boundary = current_minute % 15
        minutes_until_boundary = 15 - minutes_past_boundary

        if minutes_until_boundary == 15:
            # We're exactly on a boundary, wait for the next one
            minutes_until_boundary = 15

        # Calculate total seconds
        seconds_until = (
            (minutes_until_boundary * 60)
            - current_second
            - (current_microsecond / 1_000_000)
        )

        # Add 2-second buffer to ensure candle has closed on exchange
        buffer_seconds = 2.0

        return max(0.1, seconds_until + buffer_seconds)

    @staticmethod
    def get_next_candle_time() -> datetime:
        """Get the datetime of the next 15-minute candle close."""
        now = datetime.utcnow()
        current_minute = now.minute
        minutes_past_boundary = current_minute % 15
        minutes_until_boundary = 15 - minutes_past_boundary

        if minutes_until_boundary == 15:
            minutes_until_boundary = 15

        next_close = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_until_boundary)
        return next_close

    async def broadcast_signal(self, signal: SignalResult):
        """Broadcast signal to all connected WebSocket clients."""
        if not self.connected_clients:
            logger.debug("No clients connected for broadcast")
            return

        message = signal.model_dump_json()
        disconnected = set()

        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(client)

        # Clean up disconnected clients
        for client in disconnected:
            self.connected_clients.discard(client)

    async def compute_and_broadcast(self, skip_trade: bool = False) -> SignalResult | None:
        """Compute signals and broadcast to all clients."""
        try:
            logger.info("Computing signals...")

            # Add timeout to prevent slow TradingView from delaying trades
            # Signal must compute within 30 seconds or we skip this candle
            # Pull signal thresholds from the first running bot's config if available
            signal_kwargs = {}
            running_bots = [b for b in bot_manager.bots.values() if b.status.value == "running"]
            if running_bots:
                cfg = running_bots[0].config
                signal_kwargs = {
                    "rsi5_bull": cfg.rsi5_bull_threshold,
                    "rsi5_bear": cfg.rsi5_bear_threshold,
                    "close_pos_bull": cfg.close_pos_bull,
                    "close_pos_bear": cfg.close_pos_bear,
                    "body_ratio_min": cfg.body_ratio_min,
                    "doji_threshold": cfg.doji_threshold,
                    "volume_ratio_min": cfg.volume_ratio_min,
                    "wick_ratio_min": cfg.wick_ratio_min,
                    "consecutive_penalty": cfg.consecutive_penalty,
                    "min_score": cfg.min_score,
                    "use_engulfing": cfg.use_engulfing,
                    "mean_reversion_boost": cfg.mean_reversion_boost,
                    "use_close_position": cfg.use_close_position,
                    "use_wick_rejection": cfg.use_wick_rejection,
                    "use_body_strength": cfg.use_body_strength,
                    "use_rsi5": cfg.use_rsi5,
                    "use_volume_confirm": cfg.use_volume_confirm,
                }

            import functools
            try:
                signal = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, functools.partial(compute_signals, **signal_kwargs)
                    ),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error("Signal computation timed out after 30s - TradingView may be slow")
                # Use cached signal if available
                if self.latest_signal:
                    logger.warning("Using cached signal for auto-trade")
                    signal = self.latest_signal
                else:
                    return None

            self.latest_signal = signal
            logger.info(f"Signal computed: direction={signal.direction}, confidence={signal.confidence}")

            # Log the signal
            bot_manager.log_signal(
                direction=signal.direction,
                confidence=signal.confidence,
                score_up=signal.score_up,
                score_dn=signal.score_dn,
                price=signal.price
            )

            await self.broadcast_signal(signal)

            # Trigger auto-trading if enabled (each bot decides based on its strategy)
            if self._auto_trade_enabled and not skip_trade:
                await self._trigger_auto_trade(signal)

            return signal

        except Exception as e:
            logger.error(f"Error computing signals: {e}")
            return None

    async def _trigger_auto_trade(self, signal: SignalResult):
        """Trigger auto-trading based on the computed signal."""
        try:
            # Get current Kalshi market info
            if not self._kalshi_feed:
                logger.debug("Kalshi feed not configured, skipping auto-trade")
                return

            kalshi_price = self._kalshi_feed.get_kalshi_price()

            # At candle boundaries the old market closes before the new one appears.
            # Kalshi can take up to ~90 seconds to list the new contract.
            # Actively force the feed to scan for the new market during retries.
            retry_delay = 5
            max_wait = 120
            waited = 0

            while waited < max_wait:
                kalshi_price = self._kalshi_feed.get_kalshi_price()
                market_data = self._kalshi_feed.market_data or {}
                market_ticker = kalshi_price.market_ticker
                market_status = market_data.get("status", "")

                has_ticker = market_ticker and market_ticker != "N/A"
                is_active = market_status == "active"

                if has_ticker and is_active:
                    break  # Good to go

                # Force the feed to scan for a new market instead of passively waiting
                if hasattr(self._kalshi_feed, '_check_and_update_market'):
                    try:
                        await self._kalshi_feed._check_and_update_market()
                    except Exception as e:
                        logger.debug(f"Error forcing market scan: {e}")

                reason = "no ticker" if not has_ticker else f"status={market_status}"
                logger.info(
                    f"Kalshi market not ready ({reason}), waiting {retry_delay}s for new contract... "
                    f"({waited}/{max_wait}s)"
                )
                await asyncio.sleep(retry_delay)
                waited += retry_delay

            # Final check after retries
            kalshi_price = self._kalshi_feed.get_kalshi_price()
            market_data = self._kalshi_feed.market_data or {}

            if not kalshi_price.market_ticker or kalshi_price.market_ticker == "N/A":
                logger.warning(f"No active Kalshi market after waiting {waited}s (max {max_wait}s), skipping auto-trade")
                return

            if market_data.get("status", "") != "active":
                logger.warning(f"Market still not active after {waited}s (max {max_wait}s, status={market_data.get('status')}), skipping auto-trade")
                return

            # Check if we're within trading hours
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            open_time_str = market_data.get("open_time", "")
            close_time_str = market_data.get("close_time", "")

            if open_time_str and close_time_str:
                try:
                    open_time = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
                    close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))

                    if now < open_time:
                        logger.warning(f"Market not yet open (opens at {open_time_str}), skipping auto-trade")
                        return
                    if now > close_time:
                        logger.warning(f"Market already closed (closed at {close_time_str}), skipping auto-trade")
                        return

                    logger.info(f"Market is open: {open_time_str} to {close_time_str}")
                except Exception as e:
                    logger.debug(f"Could not parse market times: {e}")

            # Use ask price for buying (more realistic fill)
            entry_price = kalshi_price.yes_ask if kalshi_price.yes_ask > 0 else kalshi_price.price_cents

            if entry_price <= 0:
                logger.warning("Invalid Kalshi price, skipping auto-trade")
                return

            logger.info(f"Triggering auto-trade: signal={signal.direction}, prev_candle={signal.prev_candle_color} @ {entry_price}¢ on {kalshi_price.market_ticker}")

            positions = await bot_manager.on_signal(
                signal=signal,
                market_ticker=kalshi_price.market_ticker,
                market_title=kalshi_price.market_title,
                current_price_cents=entry_price
            )

            if positions:
                logger.info(f"Auto-trade opened {len(positions)} position(s)")
            else:
                logger.debug("No positions opened (no eligible running bots)")

        except Exception as e:
            logger.error(f"Error in auto-trade: {e}")

    async def run_forever(self):
        """Main scheduler loop - syncs to 15-minute candle closes."""
        self._running = True
        logger.info("Signal scheduler started")

        # Compute initial signal on startup but skip auto-trade — only trade at candle boundaries
        await self.compute_and_broadcast(skip_trade=True)

        while self._running:
            try:
                # Calculate time until next candle close
                sleep_seconds = self.seconds_until_next_15m()
                next_candle = self.get_next_candle_time()

                # Don't log if sleeping less than 10 seconds (rapid retries)
                if sleep_seconds > 10:
                    logger.info(
                        f"Sleeping {sleep_seconds:.1f}s until next candle close at {next_candle.strftime('%H:%M:%S')} UTC"
                    )

                await asyncio.sleep(sleep_seconds)

                if not self._running:
                    break

                # Compute and broadcast new signals
                await self.compute_and_broadcast()

            except asyncio.CancelledError:
                logger.info("Scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                # Wait a bit before retrying on error
                await asyncio.sleep(10)

        logger.info("Signal scheduler stopped")

    def start(self):
        """Start the scheduler as a background task."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_forever())
            logger.info("Scheduler background task started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Scheduler stopped")


# Global scheduler instance
scheduler = SignalScheduler()
