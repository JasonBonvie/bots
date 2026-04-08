"""
BTC/Kalshi 15-Minute Momentum Bot Dashboard
Main FastAPI application with REST API and WebSocket support.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from bot.signals import compute_signals
from bot.scheduler import scheduler
from bot.models import SignalResult, HealthCheck, BacktestParams, BacktestResult, OptimizeRequest, OptimizeResponse, OptimizeTrialResult, WeatherBacktestParams, WeatherBacktestResult
from bot.backtest import BacktestEngine
from bot.weather_backtest import WeatherBacktestEngine
from bot.live_price import live_price_feed
from bot.kalshi_feed import kalshi_feed
from bot.bot_manager import bot_manager, Bot, BotConfig, BotStatus, PositionSide, OrderStatus, BotLog
from bot.models import LimitTier
from bot.database import db
from pydantic import BaseModel
from typing import Optional, List
import json
import urllib.request
import time

# Simple in-memory cache for chart data (avoids rate-limiting CryptoCompare)
_chart_cache: dict = {"data": None, "ts": 0.0}
_CHART_CACHE_TTL = 60  # seconds


class BotConfigUpdate(BaseModel):
    """Request model for updating bot configuration."""
    # Strategy mode
    strategy_mode: Optional[str] = None  # "signals" or "follow_candle"

    # Entry rules (only used when strategy_mode="signals")
    min_score: Optional[float] = None
    require_ema_alignment: Optional[bool] = None
    require_rsi_zone: Optional[bool] = None
    require_macd_confirmation: Optional[bool] = None
    require_close_position: Optional[bool] = None

    # Position sizing
    base_stake_cents: Optional[int] = None
    max_position_size: Optional[int] = None
    use_martingale: Optional[bool] = None
    martingale_multiplier: Optional[float] = None
    max_martingale_level: Optional[int] = None

    # Order execution
    use_limit_orders: Optional[bool] = None
    limit_price_cents: Optional[int] = None  # 1-99 cents
    limit_order_expiry_seconds: Optional[int] = None  # 0 = no expiry
    limit_ladder: Optional[List[LimitTier]] = None  # additional limit tiers

    # Risk management
    daily_loss_limit_cents: Optional[int] = None
    daily_trade_limit: Optional[int] = None
    stop_loss_cents: Optional[int] = None
    take_profit_cents: Optional[int] = None


class CreateBotRequest(BaseModel):
    """Request model for creating a bot."""
    name: str
    description: str = ""
    config: Optional[BotConfigUpdate] = None

# Check if live trading is enabled via environment variable
LIVE_TRADING_ENABLED = os.getenv("KALSHI_LIVE_TRADING", "").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - start/stop scheduler and price feeds."""
    logger.info("Starting application...")

    # Initialize database for persistence
    logger.info("Initializing database...")
    await db.initialize()
    await bot_manager.set_database(db)
    logger.info("Database initialized and bot data loaded")

    # Start the price feeds
    await live_price_feed.start()  # Kraken for live BTC spot price
    await kalshi_feed.start()       # Kalshi for contract prices

    # Wire up Kalshi client to bot manager for order execution
    bot_manager.set_kalshi_client(
        kalshi_feed.client,
        enable_live_trading=LIVE_TRADING_ENABLED
    )
    if LIVE_TRADING_ENABLED:
        logger.warning("⚠️  LIVE TRADING ENABLED - Real orders will be placed on Kalshi!")
    else:
        logger.info("Paper trading mode - Orders will be simulated locally")

    # Connect Kalshi feed to scheduler for auto-trading
    scheduler.set_kalshi_feed(kalshi_feed)

    # Wire Kalshi price updates into bot manager so position PnL stays current
    kalshi_feed.set_price_update_callback(bot_manager.update_position_prices)

    # Set up log broadcast callback
    bot_manager.set_log_callback(broadcast_log)

    # Start background task to cleanup expired limit orders
    await bot_manager.start_cleanup_task()

    # Start the signal scheduler in background
    scheduler.start()

    yield

    # Cleanup on shutdown
    logger.info("Shutting down application...")
    await bot_manager.stop_cleanup_task()
    await live_price_feed.stop()
    await kalshi_feed.stop()
    await scheduler.stop()
    await db.close()
    logger.info("Database connection closed")


# Create FastAPI app
app = FastAPI(
    title="BTC/Kalshi Momentum Bot",
    description="15-minute momentum trading signals dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files path
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the main dashboard HTML file."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path, media_type="text/html")


@app.get("/api/signals", response_model=SignalResult)
async def get_signals():
    """
    Get the latest computed momentum signals.

    Returns the most recent signal computation, or computes fresh
    signals if none are cached.
    """
    try:
        # Return cached signal if available and recent
        if scheduler.latest_signal is not None:
            return scheduler.latest_signal

        # Otherwise compute fresh signals
        logger.info("Computing fresh signals for API request")
        signal = await asyncio.get_event_loop().run_in_executor(
            None, compute_signals
        )
        scheduler.latest_signal = signal
        return signal

    except Exception as e:
        logger.error(f"Error computing signals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute signals: {str(e)}"
        )


@app.get("/api/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint with scheduler status."""
    now = datetime.now(timezone.utc)
    next_candle = scheduler.get_next_candle_time().replace(tzinfo=timezone.utc)
    seconds_until = max(0, int((next_candle - now).total_seconds()))

    return HealthCheck(
        status="healthy",
        timestamp=now,
        next_candle_close=next_candle,
        seconds_until_close=seconds_until
    )


@app.get("/api/live_price")
async def get_live_price():
    """Get the current live BTC price from Kraken."""
    return live_price_feed.get_live_price()


@app.get("/api/kalshi_price")
async def get_kalshi_price():
    """Get the current Kalshi contract price."""
    return kalshi_feed.get_kalshi_price()


@app.get("/api/kalshi_markets")
async def get_kalshi_markets():
    """Get available Kalshi BTC markets."""
    markets = await kalshi_feed.client.search_btc_markets()
    return {"markets": markets[:20]}  # Return top 20


# =====================
# Logs API
# =====================

@app.get("/api/logs")
async def get_logs(bot_id: str = None, limit: int = 100):
    """
    Get bot activity logs.

    Args:
        bot_id: Filter by specific bot (optional)
        limit: Maximum number of logs to return (default 100)
    """
    logs = bot_manager.get_logs(bot_id=bot_id, limit=limit)
    return {"logs": [l.model_dump() for l in logs]}


async def broadcast_log(log: BotLog):
    """Broadcast a log entry to all connected WebSocket clients."""
    if not scheduler.connected_clients:
        return

    log_data = {
        "type": "bot_log",
        **log.model_dump()
    }

    # Convert datetime to ISO string for JSON serialization
    log_data["timestamp"] = log.timestamp.isoformat()

    import json
    message = json.dumps(log_data)

    disconnected = set()
    for client in scheduler.connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    for client in disconnected:
        scheduler.connected_clients.discard(client)


# =====================
# Kalshi Trading API
# =====================

@app.get("/api/kalshi/balance")
async def get_kalshi_balance():
    """Get Kalshi account balance."""
    balance = await bot_manager.get_kalshi_balance()
    if balance is None:
        return {
            "balance_cents": 0,
            "balance_dollars": 0.0,
            "payout_cents": 0,
            "authenticated": False,
            "error": "Kalshi API not configured or authentication failed"
        }
    return {
        "balance_cents": balance.get("balance", 0),
        "balance_dollars": balance.get("balance", 0) / 100.0,
        "payout_cents": balance.get("payout", 0),
        "authenticated": True
    }


@app.get("/api/kalshi/positions")
async def get_kalshi_positions():
    """Get positions directly from Kalshi API."""
    if not kalshi_feed.client:
        return {"positions": [], "error": "Kalshi client not configured"}

    positions = await kalshi_feed.client.get_positions()
    return {"positions": positions}


@app.get("/api/kalshi/orders")
async def get_kalshi_orders(status: str = None, ticker: str = None, limit: int = 50):
    """Get orders from Kalshi API."""
    if not kalshi_feed.client:
        return {"orders": [], "error": "Kalshi client not configured"}

    orders = await kalshi_feed.client.get_orders(ticker=ticker, status=status, limit=limit)
    return {"orders": orders}


@app.get("/api/kalshi/fills")
async def get_kalshi_fills(ticker: str = None, limit: int = 50):
    """Get trade fills from Kalshi API."""
    if not kalshi_feed.client:
        return {"fills": [], "error": "Kalshi client not configured"}

    fills = await kalshi_feed.client.get_fills(ticker=ticker, limit=limit)
    return {"fills": fills}


@app.post("/api/kalshi/order")
async def place_kalshi_order(
    ticker: str,
    side: str,
    action: str,
    count: int,
    order_type: str = "market",
    price_cents: int = None
):
    """
    Place an order directly on Kalshi.

    Args:
        ticker: Market ticker
        side: "yes" or "no"
        action: "buy" or "sell"
        count: Number of contracts
        order_type: "market" or "limit"
        price_cents: Price in cents (required for limit orders)
    """
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured")

    if not LIVE_TRADING_ENABLED:
        raise HTTPException(
            status_code=403,
            detail="Live trading is disabled. Set KALSHI_LIVE_TRADING=true to enable."
        )

    if order_type == "limit" and price_cents is None:
        raise HTTPException(status_code=400, detail="price_cents required for limit orders")

    result = await kalshi_feed.client.place_order(
        ticker=ticker,
        side=side,
        action=action,
        count=count,
        type=order_type,
        yes_price=price_cents if side == "yes" else None,
        no_price=price_cents if side == "no" else None
    )

    if result and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.delete("/api/kalshi/order/{order_id}")
async def cancel_kalshi_order(order_id: str):
    """Cancel a Kalshi order."""
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured")

    success = await kalshi_feed.client.cancel_order(order_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel order")

    return {"success": True, "order_id": order_id}


@app.post("/api/kalshi/sync")
async def sync_kalshi_positions():
    """Sync local positions with Kalshi API."""
    synced = await bot_manager.sync_kalshi_positions()
    return {"synced_positions": synced}


@app.get("/api/trading/status")
async def get_trading_status():
    """Get current trading system status."""
    return {
        "live_trading_enabled": LIVE_TRADING_ENABLED,
        "kalshi_client_configured": kalshi_feed.client is not None,
        "kalshi_authenticated": kalshi_feed.client._private_key is not None if kalshi_feed.client else False,
        "current_market": kalshi_feed.market_ticker,
        "bot_manager_live": bot_manager.is_live_trading,
        "auto_trade_enabled": scheduler._auto_trade_enabled
    }


@app.post("/api/trading/live_mode")
async def set_live_trading(enabled: bool):
    """
    Toggle live trading mode at runtime without restarting.
    When disabled, orders are simulated locally (paper trading).
    When enabled, real orders are placed on Kalshi.
    """
    global LIVE_TRADING_ENABLED
    LIVE_TRADING_ENABLED = enabled
    bot_manager.set_kalshi_client(kalshi_feed.client, enable_live_trading=enabled)
    return {
        "live_trading_enabled": enabled,
        "message": f"Live trading {'ENABLED — real orders will be placed' if enabled else 'DISABLED — paper trading mode'}"
    }


@app.post("/api/trading/auto_trade")
async def toggle_auto_trade(enabled: bool):
    """
    Enable or disable auto-trading.

    When enabled, running bots will automatically open positions
    when signals are computed at 15-minute candle closes.
    """
    scheduler.set_auto_trade(enabled)
    return {
        "auto_trade_enabled": enabled,
        "message": f"Auto-trading {'enabled' if enabled else 'disabled'}"
    }


@app.get("/api/chart_data")
async def get_chart_data():
    """Get historical 15-minute candlestick data for the BTC chart from Binance."""
    global _chart_cache

    # Return cached data if still fresh
    if _chart_cache["data"] is not None and (time.time() - _chart_cache["ts"]) < _CHART_CACHE_TTL:
        return _chart_cache["data"]

    try:
        def _fetch():
            url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=101"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            for attempt in range(3):
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    return data
                msg = data.get("msg", "Unknown") if isinstance(data, dict) else "Unknown"
                if attempt < 2:
                    import time as _t; _t.sleep(1.0)
                    continue
                raise ValueError(f"Binance error: {msg}")
            raise ValueError("Binance: max retries exceeded")

        raw = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        candles = []
        for k in raw:
            candles.append({
                "time": int(k[0]) // 1000,  # Binance open_time ms → seconds
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4])
            })

        result = {"candles": candles}
        _chart_cache["data"] = result
        _chart_cache["ts"] = time.time()
        return result

    except Exception as e:
        logger.error(f"Error fetching chart data from Binance: {e}")
        # Return stale cache if available rather than empty
        if _chart_cache["data"] is not None:
            logger.warning("Returning stale chart cache due to API error")
            return _chart_cache["data"]
        return {"candles": [], "error": str(e)}


# =====================
# Bot Management API
# =====================

@app.get("/api/bots")
async def get_bots():
    """Get all bots."""
    bots = await bot_manager.get_all_bots()
    summary = await bot_manager.get_summary()
    return {"bots": [b.model_dump() for b in bots], "summary": summary}


@app.post("/api/bots")
async def create_bot(request: CreateBotRequest):
    """
    Create a new bot with optional configuration.

    Configuration options:
    - min_score: Minimum signal score to enter (default: 3.0)
    - require_ema_alignment: Require EMA8 > EMA21 for UP (default: true)
    - require_rsi_zone: Require RSI in momentum zone (default: true)
    - require_macd_confirmation: Require MACD > signal line (default: true)
    - require_close_position: Require close position filter (default: false)
    - base_stake_cents: Base stake in cents (default: 100 = $1.00)
    - max_position_size: Max contracts per trade (default: 10)
    - use_martingale: Enable martingale staking (default: false)
    - martingale_multiplier: Multiplier after loss (default: 2.0)
    - max_martingale_level: Max martingale steps (default: 5)
    - daily_loss_limit_cents: Max daily loss in cents (default: 10000 = $100)
    - daily_trade_limit: Max trades per day (default: 50)
    - stop_loss_cents: Stop loss per position, 0=disabled (default: 0)
    - take_profit_cents: Take profit per position, 0=disabled (default: 0)
    """
    # Build config from request
    config = BotConfig()
    if request.config:
        for field, value in request.config.model_dump(exclude_none=True).items():
            if hasattr(config, field):
                setattr(config, field, value)

    bot = await bot_manager.create_bot(
        name=request.name,
        description=request.description,
        config=config
    )
    return bot.model_dump()


@app.get("/api/bots/{bot_id}/config")
async def get_bot_config(bot_id: str):
    """Get a bot's configuration settings."""
    bot = await bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    return {
        "bot_id": bot_id,
        "bot_name": bot.name,
        "config": bot.config.model_dump()
    }


@app.put("/api/bots/{bot_id}/config")
async def update_bot_config(bot_id: str, config_update: BotConfigUpdate):
    """
    Update a bot's configuration settings.

    Only provided fields will be updated; omitted fields keep their current values.

    Example request body:
    {
        "base_stake_cents": 200,
        "use_martingale": true,
        "daily_loss_limit_cents": 5000
    }
    """
    bot = await bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    # Update only the provided fields
    updates = config_update.model_dump(exclude_none=True)
    for field, value in updates.items():
        if hasattr(bot.config, field):
            setattr(bot.config, field, value)

    bot = await bot_manager.update_bot(bot_id, config=bot.config)
    return {
        "bot_id": bot_id,
        "bot_name": bot.name,
        "config": bot.config.model_dump(),
        "message": f"Updated {len(updates)} configuration field(s)"
    }


@app.get("/api/bots/{bot_id}")
async def get_bot(bot_id: str):
    """Get a specific bot."""
    bot = await bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    metrics = await bot_manager.get_metrics(bot_id)
    positions = await bot_manager.get_positions(bot_id)
    trades = await bot_manager.get_trades(bot_id, limit=20)
    return {
        "bot": bot.model_dump(),
        "metrics": metrics.model_dump() if metrics else None,
        "positions": [p.model_dump() for p in positions],
        "trades": [t.model_dump() for t in trades]
    }


@app.delete("/api/bots/{bot_id}")
async def delete_bot(bot_id: str):
    """Delete a bot."""
    success = await bot_manager.delete_bot(bot_id)
    if not success:
        raise HTTPException(status_code=404, detail="Bot not found")
    return {"success": True}


@app.post("/api/bots/{bot_id}/start")
async def start_bot(bot_id: str):
    """Start a bot."""
    bot = await bot_manager.set_bot_status(bot_id, BotStatus.RUNNING)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    return bot.model_dump()


@app.post("/api/bots/{bot_id}/pause")
async def pause_bot(bot_id: str):
    """Pause a bot."""
    bot = await bot_manager.set_bot_status(bot_id, BotStatus.PAUSED)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    return bot.model_dump()


@app.post("/api/bots/{bot_id}/stop")
async def stop_bot(bot_id: str):
    """Stop a bot."""
    bot = await bot_manager.set_bot_status(bot_id, BotStatus.STOPPED)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    return bot.model_dump()


@app.post("/api/bots/{bot_id}/reset_martingale")
async def reset_martingale(bot_id: str):
    """Reset martingale level to 0 for a bot."""
    bot = bot_manager.bots.get(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    prev_level = bot.current_martingale_level
    bot.current_martingale_level = 0
    await bot_manager._save_bot_to_db(bot)
    return {"bot_id": bot_id, "previous_level": prev_level, "current_level": 0}


@app.get("/api/positions")
async def get_positions(bot_id: str = None):
    """Get all open positions."""
    positions = await bot_manager.get_positions(bot_id)
    return {"positions": [p.model_dump() for p in positions]}


@app.post("/api/positions")
async def open_position(
    bot_id: str,
    market_ticker: str,
    side: str,
    entry_price_cents: int,
    quantity: int = 1
):
    """Open a new position."""
    market_title = f"BTC 15m - {market_ticker}"
    position_side = PositionSide.YES if side.lower() == "yes" else PositionSide.NO
    position = await bot_manager.open_position(
        bot_id=bot_id,
        market_ticker=market_ticker,
        market_title=market_title,
        side=position_side,
        entry_price_cents=entry_price_cents,
        quantity=quantity
    )
    if not position:
        raise HTTPException(status_code=400, detail="Could not open position")
    return position.model_dump()


@app.post("/api/positions/{position_id}/close")
async def close_position(position_id: str, exit_price_cents: int):
    """Close a position."""
    trade = await bot_manager.close_position(position_id, exit_price_cents)
    if not trade:
        raise HTTPException(status_code=404, detail="Position not found")
    return trade.model_dump()


@app.get("/api/trades")
async def get_trades(bot_id: str = None, limit: int = 50):
    """Get trade history."""
    trades = await bot_manager.get_trades(bot_id, limit)
    return {"trades": [t.model_dump() for t in trades]}


@app.post("/api/positions/{position_id}/sync")
async def sync_position_status(position_id: str):
    """
    Sync order status from Kalshi API for a specific position.

    Returns the updated position status.
    """
    position = await bot_manager.get_positions()
    position = next((p for p in position if p.id == position_id), None)
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")

    if not LIVE_TRADING_ENABLED:
        return {
            "position_id": position_id,
            "status": position.order_status.value,
            "message": "Live trading not enabled - cannot sync with Kalshi"
        }

    result = await bot_manager.sync_order_status(position_id)

    # Get updated position
    positions = await bot_manager.get_positions()
    updated_position = next((p for p in positions if p.id == position_id), None)

    return {
        "position_id": position_id,
        "status": updated_position.order_status.value if updated_position else "unknown",
        "kalshi_order_id": updated_position.kalshi_order_id if updated_position else None,
        "filled_quantity": updated_position.filled_quantity if updated_position else 0,
        "sync_result": result
    }


@app.post("/api/positions/sync")
async def sync_all_positions():
    """
    Sync order status from Kalshi API for all pending positions.

    Returns sync results for all positions.
    """
    if not LIVE_TRADING_ENABLED:
        return {
            "synced": 0,
            "message": "Live trading not enabled - cannot sync with Kalshi"
        }

    result = await bot_manager.sync_order_status()
    return result


@app.post("/api/bots/{bot_id}/auto_trade")
async def trigger_auto_trade(bot_id: str):
    """
    Manually trigger auto-trading for a specific bot based on current signal.
    Useful for testing the auto-trade logic.
    """
    bot = await bot_manager.get_bot(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    if bot.status != BotStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Bot must be running to auto-trade")

    # Get current signal
    if scheduler.latest_signal is None:
        raise HTTPException(status_code=400, detail="No signal available")

    signal = scheduler.latest_signal
    kalshi_price = kalshi_feed.get_kalshi_price()

    # Trigger auto-trade
    positions = await bot_manager.on_signal(
        direction=signal.direction,
        confidence=signal.confidence,
        market_ticker=kalshi_price.market_ticker,
        market_title=kalshi_price.market_title,
        current_price_cents=kalshi_price.yes_ask or kalshi_price.price_cents
    )

    return {
        "positions_opened": len(positions),
        "positions": [p.model_dump() for p in positions],
        "signal_direction": signal.direction,
        "signal_confidence": signal.confidence
    }


# =====================
# Backtest API
# =====================

@app.post("/api/backtest", response_model=BacktestResult)
async def run_backtest(params: BacktestParams):
    """
    Run a strategy backtest over historical BTC data.

    BTC-only mode: fast, no Kalshi API calls — direction vs next-candle outcome.
    Kalshi mode: fetches real settled markets + candlestick entry prices.
    """
    try:
        if params.mode == "kalshi":
            if not kalshi_feed.client:
                raise HTTPException(
                    status_code=400,
                    detail="Kalshi client not configured. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH."
                )
            # Pre-fetch Kalshi data async before handing off to executor
            markets = await kalshi_feed.client.get_settled_markets(
                "KXBTC15M", params.start_ts, params.end_ts
            )
            candles: dict = {}
            for m in markets:
                ticker = m.get("ticker", "")
                if ticker:
                    # Use market's own time window (15-min market + buffer)
                    try:
                        close_ts = int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp())
                        market_start = close_ts - 1200  # 20 min before close
                        market_end = close_ts + 60      # 1 min after close
                    except Exception:
                        market_start = params.start_ts
                        market_end = params.end_ts
                    candles[ticker] = await kalshi_feed.client.get_historical_candlesticks(
                        ticker, market_start, market_end
                    )
                    await asyncio.sleep(0.1)  # gentle rate limiting

            engine = BacktestEngine(params)
            df_btc = await asyncio.get_event_loop().run_in_executor(
                None, engine.fetch_btc_ohlcv
            )
            result = await asyncio.get_event_loop().run_in_executor(
                None, engine.run_kalshi_mode, df_btc, markets, candles
            )
        else:
            engine = BacktestEngine(params)
            result = await asyncio.get_event_loop().run_in_executor(
                None, engine.run_btc_only
            )

        return result

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        import traceback
        logger.error(f"Backtest error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(exc)}")


@app.get("/api/weather/candle_debug")
async def weather_candle_debug(ticker: str):
    """
    Return raw first candlestick for a weather market ticker.
    Tries multiple period_intervals and also fetches the raw market object.
    e.g. /api/weather/candle_debug?ticker=KXLOWTNYC-26APR070415-15
    """
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured.")
    from datetime import datetime, timezone, timedelta
    now      = datetime.now(timezone.utc)
    end_ts   = int(now.timestamp())
    start_ts = int((now - timedelta(days=7)).timestamp())

    results: dict = {}

    # Try multiple period intervals
    for interval in [1, 10, 60]:
        try:
            candles = await kalshi_feed.client.get_historical_candlesticks(
                ticker, start_ts=start_ts, end_ts=end_ts, period_interval=interval
            )
            results[f"period_{interval}min"] = {
                "candle_count": len(candles),
                "first_candle": candles[0] if candles else None,
            }
        except Exception as exc:
            results[f"period_{interval}min"] = {"error": str(exc)}

    # Also fetch raw market object to see what price fields are available
    try:
        await kalshi_feed.client._ensure_session()
        path = f"/trade-api/v2/markets/{ticker}"
        url = f"{kalshi_feed.client.base_url.replace('/trade-api/v2', '')}{path}"
        headers = kalshi_feed.client._get_auth_headers("GET", path)
        async with kalshi_feed.client._session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                market = data.get("market", data)
                results["market_object"] = market
                results["market_keys"] = list(market.keys()) if isinstance(market, dict) else []
            else:
                body = await resp.text()
                results["market_object"] = {"status": resp.status, "body": body[:500]}
    except Exception as exc:
        results["market_object"] = {"error": str(exc)}

    return {"ticker": ticker, **results}


@app.get("/api/weather/market_debug")
async def weather_market_debug(series: str, limit: int = 3):
    """
    Show raw settled market objects (all fields) for a series.
    Use to discover price fields available on settled weather markets.
    e.g. /api/weather/market_debug?series=KXLOWTMIA&limit=3
    """
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured.")
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    markets = await kalshi_feed.client.get_settled_markets(
        series_ticker=series,
        min_close_ts=int((now - timedelta(days=30)).timestamp()),
        max_close_ts=int(now.timestamp()),
        limit=limit,
    )
    return {
        "series": series,
        "count": len(markets),
        "markets": markets[:limit],
        "sample_keys": list(markets[0].keys()) if markets else [],
    }


@app.get("/api/weather/search")
async def search_weather_series(keyword: str = "weather"):
    """
    Search Kalshi markets by keyword and return unique series tickers.
    Use this to discover real series names before running a backtest.
    """
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured.")
    try:
        results = await kalshi_feed.client.search_series_by_keyword(keyword)
        return {"keyword": keyword, "series": results}
    except Exception as exc:
        logger.error(f"Weather search error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/backtest/weather", response_model=WeatherBacktestResult)
async def run_weather_backtest(params: WeatherBacktestParams):
    """
    Backtest a low-probability weather market betting strategy.
    Fetches all settled markets for the requested series in the date range,
    enters when opening price <= max_entry_cents, and tracks P&L at settlement.
    Requires Kalshi API credentials.
    """
    try:
        if not kalshi_feed.client:
            raise HTTPException(
                status_code=400,
                detail="Kalshi client not configured. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH."
            )
        engine = WeatherBacktestEngine(kalshi_client=kalshi_feed.client)
        return await engine.run(params)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        import traceback
        logger.error(f"Weather backtest error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Weather backtest failed: {str(exc)}")


# ── Weather Live Dashboard ──────────────────────────────────────────────────

class SingleOrderReq(BaseModel):
    ticker: str
    side: str          # "yes" | "no"
    price_cents: int   # limit price 1-99
    count: int         # contracts

class BulkOrderReq(BaseModel):
    orders: List[SingleOrderReq]
    expiry_minutes: int = 120   # cancel unfilled orders after this many minutes


@app.get("/api/weather/live-markets")
async def get_live_markets(series: str):
    """
    Fetch all currently open Kalshi markets for a comma-separated list of series tickers.
    Returns prices in cents plus derived no_ask_cents.
    e.g. /api/weather/live-markets?series=KXLOWTNYC,KXLOWTCHI,KXLOWTLAX
    """
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured.")
    tickers = [s.strip() for s in series.split(",") if s.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="Provide at least one series ticker.")
    try:
        markets = await kalshi_feed.client.get_live_markets(tickers)
        return {"count": len(markets), "markets": markets}
    except Exception as exc:
        logger.error(f"live-markets error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/weather/bulk-orders")
async def place_bulk_orders(req: BulkOrderReq):
    """
    Place multiple limit orders at once across weather markets.
    Each order specifies ticker, side, price_cents, and count.
    Orders expire after req.expiry_minutes if unfilled.
    """
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured.")

    from datetime import datetime, timezone, timedelta
    expiry_ts = int((datetime.now(timezone.utc) + timedelta(minutes=req.expiry_minutes)).timestamp())

    results = []
    placed = 0
    failed = 0

    for o in req.orders:
        price = max(1, min(99, o.price_cents))
        count = max(1, o.count)
        side = o.side.lower()

        kwargs = dict(
            ticker=o.ticker,
            side=side,
            action="buy",
            count=count,
            type="limit",
            expiration_ts=expiry_ts,
        )
        if side == "yes":
            kwargs["yes_price"] = price
        else:
            kwargs["no_price"] = price

        result = await kalshi_feed.client.place_order(**kwargs)
        if result and "error" not in result:
            placed += 1
            results.append({"ticker": o.ticker, "side": side, "price": price,
                            "count": count, "status": "placed",
                            "order_id": result.get("order_id", "")})
        else:
            failed += 1
            err = result.get("error", "unknown") if result else "no response"
            results.append({"ticker": o.ticker, "side": side, "price": price,
                            "count": count, "status": "failed", "error": err})

    return {"placed": placed, "failed": failed, "results": results}


@app.get("/api/weather/open-orders")
async def get_open_weather_orders():
    """Return all resting (unfilled) orders from the portfolio."""
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured.")
    try:
        orders = await kalshi_feed.client.get_orders(status="resting", limit=200)
        return {"count": len(orders), "orders": orders}
    except Exception as exc:
        logger.error(f"open-orders error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/weather/orders/{order_id}")
async def cancel_weather_order(order_id: str):
    """Cancel a single open order by ID."""
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured.")
    ok = await kalshi_feed.client.cancel_order(order_id)
    return {"cancelled": ok, "order_id": order_id}


@app.post("/api/weather/cancel-all-orders")
async def cancel_all_weather_orders():
    """Cancel ALL currently resting orders in the portfolio."""
    if not kalshi_feed.client:
        raise HTTPException(status_code=400, detail="Kalshi client not configured.")
    try:
        # Fetch all resting orders then cancel each (batch endpoint unreliable)
        orders = await kalshi_feed.client.get_orders(status="resting", limit=500)
        cancelled = 0
        errors = []
        for o in orders:
            oid = o.get("order_id") or o.get("id", "")
            if not oid:
                continue
            ok = await kalshi_feed.client.cancel_order(oid)
            if ok:
                cancelled += 1
            else:
                errors.append(oid)
        return {"cancelled": cancelled, "errors": errors}
    except Exception as exc:
        logger.error(f"cancel-all-orders error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time signal and price updates.

    Clients connect here to receive:
    - Live BTC price updates every 500ms (from Kraken)
    - Kalshi contract price updates every 500ms
    - Signal broadcasts every 15 minutes when a new candle closes
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected from {websocket.client}")

    # Add to all broadcast lists
    scheduler.add_client(websocket)
    live_price_feed.add_client(websocket)
    kalshi_feed.add_client(websocket)

    try:
        # Send current signal immediately if available
        if scheduler.latest_signal is not None:
            await websocket.send_text(scheduler.latest_signal.model_dump_json())

        # Send current live BTC price if available
        if live_price_feed.current_price > 0:
            await websocket.send_text(live_price_feed.get_live_price().model_dump_json())

        # Send current Kalshi price if available
        if kalshi_feed.current_price_cents > 0:
            await websocket.send_text(kalshi_feed.get_kalshi_price().model_dump_json())

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for any message (ping/pong or refresh request)
                data = await websocket.receive_text()

                # If client requests refresh, send current signal
                if data == "refresh":
                    logger.info("Client requested manual refresh")
                    signal = await asyncio.get_event_loop().run_in_executor(
                        None, compute_signals
                    )
                    scheduler.latest_signal = signal
                    await websocket.send_text(signal.model_dump_json())

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        # Remove from all broadcast lists
        scheduler.remove_client(websocket)
        live_price_feed.remove_client(websocket)
        kalshi_feed.remove_client(websocket)
        logger.info(f"WebSocket client disconnected from {websocket.client}")


@app.get("/api/refresh", response_model=SignalResult)
async def refresh_signals():
    """
    Force refresh signals immediately.

    This endpoint computes fresh signals regardless of cache,
    useful for the manual refresh button.
    """
    try:
        logger.info("Manual refresh requested")
        signal = await asyncio.get_event_loop().run_in_executor(
            None, compute_signals
        )
        scheduler.latest_signal = signal

        # Also broadcast to all WebSocket clients
        await scheduler.broadcast_signal(signal)

        return signal

    except Exception as e:
        logger.error(f"Error refreshing signals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh signals: {str(e)}"
        )


# =====================
# Strategy Optimizer API
# =====================

@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize_strategy(req: OptimizeRequest):
    """
    Random-search optimizer: runs up to `trials` backtests with randomly
    sampled parameter combinations and returns the top N ranked by `metric`.

    Only btc_only mode is supported (fast, no Kalshi API calls needed).
    """
    import random
    import time as _time

    MAX_TRIALS = 300
    trials = min(req.trials, MAX_TRIALS)
    t0 = _time.time()

    # ── parameter search space ─────────────────────────────────────────────
    def sample_params() -> BacktestParams:
        p = BacktestParams(
            start_ts=req.start_ts,
            end_ts=req.end_ts,
            mode="btc_only",
            stake_cents=req.stake_cents,
            entry_price_cents=req.entry_price_cents,
            slippage_cents=req.slippage_cents,
        )

        if req.vary_min_score:
            p.min_score = random.randint(1, 4)

        if req.vary_signals:
            # Each signal has an independent 70% chance of being enabled.
            # Ensure at least 2 signals are on so we get real trades.
            for _ in range(20):  # retry until we get at least 2 enabled
                p.use_close_position = random.random() > 0.3
                p.use_wick_rejection  = random.random() > 0.3
                p.use_body_strength   = random.random() > 0.3
                p.use_rsi5            = random.random() > 0.3
                p.use_volume_confirm  = random.random() > 0.3
                p.use_engulfing       = random.random() > 0.3
                p.mean_reversion_boost = random.random() > 0.4
                enabled = sum([
                    p.use_close_position, p.use_wick_rejection, p.use_body_strength,
                    p.use_rsi5, p.use_volume_confirm, p.use_engulfing,
                ])
                if enabled >= 2:
                    break

        if req.vary_thresholds:
            p.consecutive_penalty = random.randint(2, 8)
            p.body_ratio_min      = round(random.uniform(0.35, 0.75), 2)
            p.wick_ratio_min      = round(random.uniform(0.15, 0.55), 2)
            p.volume_ratio_min    = round(random.uniform(0.9, 2.2), 2)
            p.close_pos_bull      = round(random.uniform(0.55, 0.75), 2)
            p.close_pos_bear      = round(1.0 - p.close_pos_bull, 2)
            p.rsi5_bull           = round(random.uniform(50, 65), 1)
            p.rsi5_bear           = round(random.uniform(35, 50), 1)

        if req.vary_martingale:
            p.use_martingale       = random.random() > 0.5
            p.martingale_multiplier = round(random.uniform(1.5, 3.0), 1)
            p.max_martingale_level = random.randint(3, 7)

        return p

    # ── run trials in executor (CPU-bound) ────────────────────────────────
    def run_all_trials():
        results = []
        # Pre-fetch BTC OHLCV once and reuse across all trials
        seed_params = BacktestParams(
            start_ts=req.start_ts, end_ts=req.end_ts,
            mode="btc_only",
            stake_cents=req.stake_cents, entry_price_cents=req.entry_price_cents,
        )
        seed_engine = BacktestEngine(seed_params)
        try:
            df_btc = seed_engine.fetch_btc_ohlcv()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch BTC data: {e}")

        for trial_idx in range(trials):
            p = sample_params()
            try:
                engine = BacktestEngine(p)
                result = engine.run_btc_only(df_btc=df_btc)
            except Exception:
                continue

            if result.trades_taken < req.min_trades:
                continue

            metric_val = getattr(result, req.metric, None)
            if metric_val is None or (isinstance(metric_val, float) and (metric_val != metric_val)):
                continue

            results.append((metric_val, trial_idx, result, p))

        return results

    raw = await asyncio.get_event_loop().run_in_executor(None, run_all_trials)

    # ── sort descending by metric, take top_n ─────────────────────────────
    raw.sort(key=lambda x: x[0], reverse=True)
    top = raw[: req.top_n]

    trial_results = []
    for rank, (metric_val, trial_idx, r, p) in enumerate(top, start=1):
        trial_results.append(OptimizeTrialResult(
            rank=rank,
            trial=trial_idx + 1,
            trades=r.trades_taken,
            win_rate=r.win_rate,
            total_pnl_cents=r.total_pnl_cents,
            max_drawdown_cents=r.max_drawdown_cents,
            sharpe_ratio=r.sharpe_ratio,
            sortino_ratio=r.sortino_ratio,
            profit_factor=min(r.profit_factor, 99.0),
            calmar_ratio=r.calmar_ratio,
            expectancy=r.expectancy,
            params=p,
        ))

    return OptimizeResponse(
        trials_run=trials,
        trials_valid=len(raw),
        metric=req.metric,
        duration_seconds=round(_time.time() - t0, 2),
        results=trial_results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
