"""
Bot Manager module for tracking bots, positions, and metrics.

Provides persistent storage for bot configurations, open positions,
trade history, and performance metrics using SQLite.

Integrates with Kalshi API for real order execution.
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, TYPE_CHECKING, Callable, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging

from .models import LimitTier

if TYPE_CHECKING:
    from .kalshi_feed import KalshiClient
    from .database import Database
    from .models import SignalResult

logger = logging.getLogger(__name__)


class BotStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class PositionSide(str, Enum):
    YES = "yes"
    NO = "no"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class LogLevel(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    TRADE = "trade"
    SIGNAL = "signal"


class BotLog(BaseModel):
    """A log entry for bot activity."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bot_id: Optional[str] = None
    bot_name: Optional[str] = None
    level: LogLevel = LogLevel.INFO
    action: str  # e.g., "bot_created", "position_opened", "signal_received"
    message: str
    details: Optional[Dict] = None


class Position(BaseModel):
    """An open position in a Kalshi market."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    bot_id: str
    market_ticker: str
    market_title: str
    side: PositionSide
    entry_price_cents: int
    quantity: int
    entry_time: datetime
    current_price_cents: int = 0
    unrealized_pnl_cents: int = 0
    # Kalshi order tracking
    kalshi_order_id: Optional[str] = None
    order_status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    rejection_reason: Optional[str] = None  # Why order was rejected (if applicable)
    # Limit order expiry tracking
    limit_order_expiry: Optional[datetime] = None  # When the limit order expires
    is_limit_order: bool = False
    # Ladder order tracking (one entry per tier, parallel to config.limit_ladder)
    ladder_order_ids: List[str] = []      # Kalshi order IDs for each ladder tier
    ladder_filled: List[bool] = []        # which tiers filled
    ladder_fill_prices: List[int] = []    # actual fill price per tier (cents)

    def update_pnl(self, current_price_cents: int):
        """Update unrealized PnL based on current price."""
        self.current_price_cents = current_price_cents
        if self.side == PositionSide.YES:
            self.unrealized_pnl_cents = (current_price_cents - self.entry_price_cents) * self.quantity
        else:
            # NO position profits when price goes down
            self.unrealized_pnl_cents = (self.entry_price_cents - current_price_cents) * self.quantity


class Trade(BaseModel):
    """A completed trade."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    bot_id: str
    market_ticker: str
    market_title: str
    side: PositionSide
    entry_price_cents: int
    exit_price_cents: int
    quantity: int
    entry_time: datetime
    exit_time: datetime
    pnl_cents: int
    result: str  # "win", "loss", "breakeven"


class BotMetrics(BaseModel):
    """Performance metrics for a bot."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0
    total_pnl_cents: int = 0
    total_pnl_dollars: float = 0.0
    avg_win_cents: float = 0.0
    avg_loss_cents: float = 0.0
    largest_win_cents: int = 0
    largest_loss_cents: int = 0
    current_streak: int = 0  # Positive for wins, negative for losses
    best_streak: int = 0
    worst_streak: int = 0


class BotConfig(BaseModel):
    """Bot configuration settings."""
    # Strategy mode
    # "signals" = use 4-filter scoring system (default)
    # "follow_candle" = bet same direction as previous candle color
    strategy_mode: str = "signals"

    # Entry rules (only used when strategy_mode="signals")
    min_score: float = 3.0          # Minimum score (0-5) to enter
    rsi5_bull_threshold: float = 55.0   # RSI(5) above this = bullish momentum
    rsi5_bear_threshold: float = 45.0   # RSI(5) below this = bearish momentum
    close_pos_bull: float = 0.65        # Close position above this = bullish
    close_pos_bear: float = 0.35        # Close position below this = bearish
    body_ratio_min: float = 0.55        # Minimum body ratio for strong candle signal
    doji_threshold: float = 0.15        # Body ratio below this = doji, skip trade
    volume_ratio_min: float = 1.3       # Volume ratio above this = high volume confirm
    wick_ratio_min: float = 0.3         # Minimum wick ratio to count as rejection
    consecutive_penalty: int = 4        # Consecutive same-color candles before mean reversion penalty
    use_close_position: bool = True      # Signal 1: close position
    use_wick_rejection: bool = True      # Signal 2: wick rejection
    use_body_strength: bool = True       # Signal 3: body strength
    use_rsi5: bool = True                # Signal 4: RSI(5)
    use_volume_confirm: bool = True      # Signal 5: volume confirm
    use_engulfing: bool = True           # +1 when candle body fully swallows previous body
    mean_reversion_boost: bool = True    # +1 to opposite direction on streak (not just penalty)
    # Context signal toggles (alpha research additions)
    use_mtf_filter: bool = False         # +1 when 1h EMA(21) + 4h EMA(50) agree with direction
    use_funding_filter: bool = False     # +1 when perpetual funding is extreme (fade the crowd)
    use_vwap_signal: bool = False        # +1 when price extended from session VWAP
    vwap_dev_pct: float = 1.5           # % deviation from VWAP to trigger signal
    # Legacy filter toggles (kept for follow_candle mode compatibility)
    require_ema_alignment: bool = False
    require_rsi_zone: bool = True
    require_macd_confirmation: bool = False
    require_close_position: bool = True

    # Position sizing
    base_stake_cents: int = 100  # $1.00 default
    max_position_size: int = 10  # Max contracts per trade
    use_martingale: bool = False
    martingale_multiplier: float = 2.0
    max_martingale_level: int = 5

    # Order execution
    use_limit_orders: bool = False  # Use limit orders instead of market orders
    limit_price_cents: int = 45  # Limit order price (1-99 cents)
    limit_order_expiry_seconds: int = 120  # 2 minutes (get filled quickly or cancel)
    limit_ladder: List[LimitTier] = []  # additional limit tiers at lower prices

    # Risk management
    daily_loss_limit_cents: int = 10000  # $100 daily loss limit
    daily_trade_limit: int = 50
    stop_loss_cents: int = 0  # 0 = no stop loss
    take_profit_cents: int = 0  # 0 = no take profit


class Bot(BaseModel):
    """A trading bot configuration and state."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    status: BotStatus = BotStatus.STOPPED
    config: BotConfig = Field(default_factory=BotConfig)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_trade_at: Optional[datetime] = None

    # Daily tracking (resets each day)
    daily_pnl_cents: int = 0
    daily_trades: int = 0
    current_martingale_level: int = 0


class BotManager:
    """
    Manages bots, positions, and trade history.

    Integrates with Kalshi API for real order execution when configured.
    Uses SQLite for persistent storage across server restarts.
    """

    def __init__(self):
        self.bots: Dict[str, Bot] = {}
        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.trades: List[Trade] = []
        self.metrics: Dict[str, BotMetrics] = {}  # bot_id -> BotMetrics
        self.logs: List[BotLog] = []  # Activity logs
        self._max_logs: int = 500  # Keep last 500 logs
        self._lock = asyncio.Lock()

        # Kalshi client for order execution (set via set_kalshi_client)
        self._kalshi_client: Optional["KalshiClient"] = None
        self._live_trading_enabled: bool = False

        # Callback for broadcasting logs (set by main.py)
        self._log_broadcast_callback: Optional[Callable] = None

        # Database for persistence (set via set_database)
        self._db: Optional["Database"] = None
        self._db_initialized: bool = False

    async def set_database(self, db: "Database"):
        """
        Set the database for persistence and load existing data.

        Args:
            db: Database instance (already initialized)
        """
        self._db = db
        await self._load_from_database()
        self._db_initialized = True
        logger.info(f"Database connected. Loaded {len(self.bots)} bots, {len(self.positions)} positions, {len(self.trades)} trades")

    async def _load_from_database(self):
        """Load all data from database on startup."""
        if not self._db:
            return

        try:
            # Load bots
            bot_models = await self._db.get_all_bots()
            for bm in bot_models:
                try:
                    config_dict = bm.get_config()
                    config = BotConfig(**config_dict) if config_dict else BotConfig()
                except Exception as cfg_err:
                    logger.warning(f"Bot {bm.id} ({bm.name}): failed to load config ({cfg_err}), using defaults")
                    config = BotConfig()
                bot = Bot(
                    id=bm.id,
                    name=bm.name,
                    description=bm.description or "",
                    status=BotStatus(bm.status) if bm.status else BotStatus.STOPPED,
                    config=config,
                    created_at=bm.created_at,
                    updated_at=bm.updated_at,
                    last_trade_at=bm.last_trade_at,
                    daily_pnl_cents=bm.daily_pnl_cents or 0,
                    daily_trades=bm.daily_trades or 0,
                    current_martingale_level=bm.current_martingale_level or 0
                )
                self.bots[bot.id] = bot
                self.metrics[bot.id] = BotMetrics()

            # Load positions
            position_models = await self._db.get_positions()
            for pm in position_models:
                position = Position(
                    id=pm.id,
                    bot_id=pm.bot_id,
                    market_ticker=pm.market_ticker,
                    market_title=pm.market_title or "",
                    side=PositionSide(pm.side),
                    entry_price_cents=pm.entry_price_cents,
                    quantity=pm.quantity,
                    entry_time=pm.entry_time,
                    current_price_cents=pm.current_price_cents or 0,
                    unrealized_pnl_cents=pm.unrealized_pnl_cents or 0,
                    kalshi_order_id=pm.kalshi_order_id,
                    order_status=OrderStatus(pm.order_status) if pm.order_status else OrderStatus.PENDING,
                    filled_quantity=pm.filled_quantity or 0,
                    rejection_reason=pm.rejection_reason,
                    limit_order_expiry=pm.limit_order_expiry,
                    is_limit_order=pm.is_limit_order or False
                )
                self.positions[position.id] = position

            # Load trades
            trade_models = await self._db.get_trades(limit=500)
            for tm in trade_models:
                trade = Trade(
                    id=tm.id,
                    bot_id=tm.bot_id,
                    market_ticker=tm.market_ticker,
                    market_title=tm.market_title or "",
                    side=PositionSide(tm.side),
                    entry_price_cents=tm.entry_price_cents,
                    exit_price_cents=tm.exit_price_cents,
                    quantity=tm.quantity,
                    entry_time=tm.entry_time,
                    exit_time=tm.exit_time,
                    pnl_cents=tm.pnl_cents,
                    result=tm.result
                )
                self.trades.append(trade)

            # Load metrics
            for bot_id in self.bots:
                metrics_model = await self._db.get_metrics(bot_id)
                if metrics_model:
                    self.metrics[bot_id] = BotMetrics(
                        total_trades=metrics_model.total_trades,
                        winning_trades=metrics_model.winning_trades,
                        losing_trades=metrics_model.losing_trades,
                        breakeven_trades=metrics_model.breakeven_trades,
                        win_rate=metrics_model.win_rate,
                        total_pnl_cents=metrics_model.total_pnl_cents,
                        avg_win_cents=metrics_model.avg_win_cents,
                        avg_loss_cents=metrics_model.avg_loss_cents,
                        largest_win_cents=metrics_model.largest_win_cents,
                        largest_loss_cents=metrics_model.largest_loss_cents,
                        current_streak=metrics_model.current_streak,
                        best_streak=metrics_model.best_streak,
                        worst_streak=metrics_model.worst_streak
                    )

            # Load recent logs
            log_models = await self._db.get_logs(limit=200)
            for lm in log_models:
                log = BotLog(
                    id=lm.id,
                    timestamp=lm.timestamp,
                    bot_id=lm.bot_id,
                    bot_name=lm.bot_name,
                    level=LogLevel(lm.level) if lm.level else LogLevel.INFO,
                    action=lm.action,
                    message=lm.message,
                    details=lm.get_details()
                )
                self.logs.append(log)

            # Reverse logs so most recent is last (for in-memory ordering)
            self.logs.reverse()

        except Exception as e:
            logger.error(f"Error loading from database: {e}")

    async def _save_bot_to_db(self, bot: Bot):
        """Save a bot to the database."""
        if not self._db:
            return
        try:
            await self._db.save_bot({
                "id": bot.id,
                "name": bot.name,
                "description": bot.description,
                "status": bot.status.value,
                "created_at": bot.created_at,
                "updated_at": bot.updated_at,
                "last_trade_at": bot.last_trade_at,
                "daily_pnl_cents": bot.daily_pnl_cents,
                "daily_trades": bot.daily_trades,
                "current_martingale_level": bot.current_martingale_level,
                "config": bot.config.model_dump()
            })
        except Exception as e:
            logger.error(f"Error saving bot to database: {e}")

    async def _save_position_to_db(self, position: Position):
        """Save a position to the database."""
        if not self._db:
            return
        try:
            await self._db.save_position({
                "id": position.id,
                "bot_id": position.bot_id,
                "market_ticker": position.market_ticker,
                "market_title": position.market_title,
                "side": position.side.value,
                "entry_price_cents": position.entry_price_cents,
                "quantity": position.quantity,
                "entry_time": position.entry_time,
                "current_price_cents": position.current_price_cents,
                "unrealized_pnl_cents": position.unrealized_pnl_cents,
                "kalshi_order_id": position.kalshi_order_id,
                "order_status": position.order_status.value,
                "filled_quantity": position.filled_quantity,
                "limit_order_expiry": position.limit_order_expiry,
                "is_limit_order": position.is_limit_order,
                "rejection_reason": position.rejection_reason
            })
        except Exception as e:
            logger.error(f"Error saving position to database: {e}")

    async def _save_trade_to_db(self, trade: Trade):
        """Save a trade to the database."""
        if not self._db:
            return
        try:
            await self._db.save_trade({
                "id": trade.id,
                "bot_id": trade.bot_id,
                "market_ticker": trade.market_ticker,
                "market_title": trade.market_title,
                "side": trade.side.value,
                "entry_price_cents": trade.entry_price_cents,
                "exit_price_cents": trade.exit_price_cents,
                "quantity": trade.quantity,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "pnl_cents": trade.pnl_cents,
                "result": trade.result
            })
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")

    async def _save_metrics_to_db(self, bot_id: str):
        """Save bot metrics to the database."""
        if not self._db or bot_id not in self.metrics:
            return
        try:
            metrics = self.metrics[bot_id]
            # Exclude computed field total_pnl_dollars which isn't in the DB model
            metrics_data = metrics.model_dump(exclude={"total_pnl_dollars"})
            await self._db.save_metrics(bot_id, metrics_data)
        except Exception as e:
            logger.error(f"Error saving metrics to database: {e}")

    async def _save_log_to_db(self, log: BotLog):
        """Save a log entry to the database."""
        if not self._db:
            return
        try:
            await self._db.save_log({
                "id": log.id,
                "timestamp": log.timestamp,
                "bot_id": log.bot_id,
                "bot_name": log.bot_name,
                "level": log.level.value,
                "action": log.action,
                "message": log.message,
                "details": log.details
            })
        except Exception as e:
            logger.error(f"Error saving log to database: {e}")

    def set_kalshi_client(self, client: "KalshiClient", enable_live_trading: bool = False):
        """
        Set the Kalshi client for order execution.

        Args:
            client: KalshiClient instance with API credentials
            enable_live_trading: If True, actually execute orders on Kalshi
        """
        self._kalshi_client = client
        self._live_trading_enabled = enable_live_trading
        logger.info(f"Kalshi client configured. Live trading: {enable_live_trading}")
        self._add_log(
            level=LogLevel.INFO,
            action="system_init",
            message=f"Trading system initialized. Live trading: {'ENABLED' if enable_live_trading else 'DISABLED (Paper Mode)'}"
        )

    def set_log_callback(self, callback: Callable):
        """Set callback for broadcasting logs via WebSocket."""
        self._log_broadcast_callback = callback

    def _add_log(
        self,
        level: LogLevel,
        action: str,
        message: str,
        bot_id: Optional[str] = None,
        bot_name: Optional[str] = None,
        details: Optional[Dict] = None
    ) -> BotLog:
        """Add a log entry and optionally broadcast it."""
        log = BotLog(
            bot_id=bot_id,
            bot_name=bot_name,
            level=level,
            action=action,
            message=message,
            details=details
        )
        self.logs.append(log)

        # Trim logs if exceeding max
        if len(self.logs) > self._max_logs:
            self.logs = self.logs[-self._max_logs:]

        # Save to database
        if self._db:
            try:
                asyncio.create_task(self._save_log_to_db(log))
            except RuntimeError:
                pass  # No event loop running

        # Broadcast log if callback is set
        if self._log_broadcast_callback:
            try:
                asyncio.create_task(self._log_broadcast_callback(log))
            except RuntimeError:
                pass  # No event loop running

        return log

    def get_logs(self, bot_id: Optional[str] = None, limit: int = 100) -> List[BotLog]:
        """Get recent logs, optionally filtered by bot_id."""
        logs = self.logs
        if bot_id:
            logs = [l for l in logs if l.bot_id == bot_id]
        return list(reversed(logs[-limit:]))  # Most recent first

    @property
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled and client is configured."""
        return self._live_trading_enabled and self._kalshi_client is not None

    # =====================
    # Bot CRUD Operations
    # =====================

    async def create_bot(self, name: str, description: str = "", config: Optional[BotConfig] = None) -> Bot:
        """Create a new bot."""
        async with self._lock:
            bot = Bot(
                name=name,
                description=description,
                config=config or BotConfig()
            )
            self.bots[bot.id] = bot
            self.metrics[bot.id] = BotMetrics()
            logger.info(f"Created bot: {bot.name} ({bot.id})")

            # Save to database
            await self._save_bot_to_db(bot)
            await self._save_metrics_to_db(bot.id)

            self._add_log(
                level=LogLevel.SUCCESS,
                action="bot_created",
                message=f"Bot '{bot.name}' created",
                bot_id=bot.id,
                bot_name=bot.name,
                details={
                    "config": {
                        "min_score": bot.config.min_score,
                        "use_limit_orders": bot.config.use_limit_orders,
                        "limit_price_cents": bot.config.limit_price_cents,
                        "base_stake_cents": bot.config.base_stake_cents
                    }
                }
            )
            return bot

    async def get_bot(self, bot_id: str) -> Optional[Bot]:
        """Get a bot by ID."""
        return self.bots.get(bot_id)

    async def get_all_bots(self) -> List[Bot]:
        """Get all bots."""
        return list(self.bots.values())

    async def update_bot(self, bot_id: str, **kwargs) -> Optional[Bot]:
        """Update a bot's configuration."""
        async with self._lock:
            bot = self.bots.get(bot_id)
            if not bot:
                return None

            for key, value in kwargs.items():
                if hasattr(bot, key):
                    setattr(bot, key, value)

            bot.updated_at = datetime.now(timezone.utc)

            # Save to database
            await self._save_bot_to_db(bot)

            return bot

    async def delete_bot(self, bot_id: str) -> bool:
        """Delete a bot and its associated data."""
        async with self._lock:
            if bot_id not in self.bots:
                return False

            bot = self.bots[bot_id]
            bot_name = bot.name

            # Close any open positions
            positions_to_close = [p for p in self.positions.values() if p.bot_id == bot_id]
            for pos in positions_to_close:
                del self.positions[pos.id]

            del self.bots[bot_id]
            if bot_id in self.metrics:
                del self.metrics[bot_id]

            logger.info(f"Deleted bot: {bot_id}")

            # Delete from database
            if self._db:
                await self._db.delete_bot(bot_id)

            self._add_log(
                level=LogLevel.WARNING,
                action="bot_deleted",
                message=f"Bot '{bot_name}' deleted",
                bot_id=bot_id,
                bot_name=bot_name,
                details={"positions_closed": len(positions_to_close)}
            )
            return True

    async def set_bot_status(self, bot_id: str, status: BotStatus) -> Optional[Bot]:
        """Set a bot's status (running, paused, stopped)."""
        async with self._lock:
            bot = self.bots.get(bot_id)
            if not bot:
                return None

            old_status = bot.status
            bot.status = status
            bot.updated_at = datetime.now(timezone.utc)
            logger.info(f"Bot {bot.name} status changed to {status}")

            # Save to database
            await self._save_bot_to_db(bot)

            status_emoji = {"running": "▶️", "paused": "⏸️", "stopped": "⏹️"}
            level = LogLevel.SUCCESS if status == BotStatus.RUNNING else LogLevel.INFO

            self._add_log(
                level=level,
                action=f"bot_{status.value}",
                message=f"{status_emoji.get(status.value, '')} Bot '{bot.name}' {status.value}",
                bot_id=bot.id,
                bot_name=bot.name,
                details={"old_status": old_status.value, "new_status": status.value}
            )
            return bot

    # =====================
    # Position Management
    # =====================

    async def open_position(
        self,
        bot_id: str,
        market_ticker: str,
        market_title: str,
        side: PositionSide,
        entry_price_cents: int,
        quantity: int,
        use_limit_order: bool = False
    ) -> Optional[Position]:
        """
        Open a new position, optionally executing on Kalshi.

        Args:
            bot_id: Bot that owns this position
            market_ticker: Kalshi market ticker
            market_title: Human-readable market title
            side: YES or NO
            entry_price_cents: Entry price in cents
            quantity: Number of contracts
            use_limit_order: If True, use limit order at entry_price_cents

        Returns:
            Position object if successful
        """
        async with self._lock:
            bot = self.bots.get(bot_id)
            if not bot or bot.status != BotStatus.RUNNING:
                logger.warning(f"Cannot open position: bot {bot_id} not running")
                return None

            # Guard: don't open a second position for the same market ticker
            duplicate = any(
                p.bot_id == bot_id and p.market_ticker == market_ticker
                for p in self.positions.values()
            )
            if duplicate:
                logger.warning(f"Bot {bot.name}: already has a position in {market_ticker}, skipping")
                return None

            # Check risk limits
            if not await self._check_risk_limits(bot, entry_price_cents * quantity):
                logger.warning(f"Position blocked by risk limits for bot {bot.name}")
                return None

            # Determine order type and price from bot config
            bot_config = bot.config
            should_use_limit = use_limit_order or bot_config.use_limit_orders
            limit_price = bot_config.limit_price_cents if bot_config.use_limit_orders else entry_price_cents

            # Validate limit price is in valid range (1-99 cents)
            if should_use_limit:
                if limit_price < 1 or limit_price > 99:
                    logger.error(f"Invalid limit price: {limit_price}¢ (must be 1-99)")
                    return None
                logger.info(f"Using limit order: price={limit_price}¢, quantity={quantity}, side={side.value}, ticker={market_ticker}")

            position = Position(
                bot_id=bot_id,
                market_ticker=market_ticker,
                market_title=market_title,
                side=side,
                entry_price_cents=limit_price if should_use_limit else entry_price_cents,
                quantity=quantity,
                entry_time=datetime.now(timezone.utc),
                current_price_cents=limit_price if should_use_limit else entry_price_cents,
                order_status=OrderStatus.PENDING
            )

            # Execute on Kalshi if live trading enabled
            if self.is_live_trading:
                # Calculate expiration time for limit orders
                expiration_ts = None
                if should_use_limit and bot_config.limit_order_expiry_seconds > 0:
                    expiration_ts = int(datetime.now(timezone.utc).timestamp() + bot_config.limit_order_expiry_seconds)

                # Log full order details before execution
                logger.info(f"Kalshi order details: ticker={market_ticker}, side={side.value}, action=buy, count={quantity}, type={'limit' if should_use_limit else 'market'}, price={limit_price if should_use_limit else 'N/A'}¢, expiry={expiration_ts}")

                order_result = await self._execute_kalshi_order(
                    ticker=market_ticker,
                    side=side.value,
                    action="buy",
                    count=quantity,
                    price_cents=limit_price if should_use_limit else None,
                    order_type="limit" if should_use_limit else "market",
                    expiration_ts=expiration_ts
                )

                if order_result and "error" not in order_result:
                    position.kalshi_order_id = order_result.get("order_id")
                    position.order_status = OrderStatus.FILLED if order_result.get("status") == "executed" else OrderStatus.PENDING
                    position.filled_quantity = order_result.get("count_executed", 0)
                    # Update entry price if filled at different price
                    if order_result.get("yes_price"):
                        position.entry_price_cents = order_result.get("yes_price")
                    logger.info(f"Kalshi order placed: {position.kalshi_order_id}")
                else:
                    # Extract detailed rejection reason
                    error_msg = "Unknown error"
                    if order_result:
                        error_msg = order_result.get("error", "Unknown error")
                        # Kalshi often returns more details in nested structure
                        if isinstance(error_msg, dict):
                            error_msg = error_msg.get("message", str(error_msg))
                        status_code = order_result.get("status_code", "N/A")
                        error_msg = f"{error_msg} (status: {status_code})"
                    else:
                        error_msg = "No response from Kalshi API"

                    logger.error(f"Failed to place Kalshi order: {error_msg}")
                    position.order_status = OrderStatus.REJECTED
                    position.rejection_reason = error_msg

                    # Log the rejection with details
                    self._add_log(
                        level=LogLevel.ERROR,
                        action="order_rejected",
                        message=f"⛔ Order REJECTED: {error_msg}",
                        bot_id=bot_id,
                        bot_name=bot.name,
                        details={
                            "market_ticker": market_ticker,
                            "side": side.value,
                            "quantity": quantity,
                            "price_cents": limit_price if should_use_limit else entry_price_cents,
                            "order_type": "limit" if should_use_limit else "market",
                            "rejection_reason": error_msg,
                            "kalshi_response": order_result
                        }
                    )
                    # Don't create local position for rejected orders in live mode
                    return None
            else:
                # Paper trading
                if should_use_limit:
                    # Limit orders stay PENDING until filled or expired
                    position.order_status = OrderStatus.PENDING
                    position.filled_quantity = 0
                else:
                    # Market orders fill immediately in paper mode
                    position.order_status = OrderStatus.FILLED
                    position.filled_quantity = quantity

            # Set limit order expiry time if using limit orders
            if should_use_limit and bot_config.limit_order_expiry_seconds > 0:
                position.limit_order_expiry = datetime.now(timezone.utc) + timedelta(seconds=bot_config.limit_order_expiry_seconds)
                position.is_limit_order = True

            # ── Limit ladder: place additional orders at lower prices ─────────
            if should_use_limit and bot_config.limit_ladder:
                position.ladder_order_ids = []
                position.ladder_filled = [False] * len(bot_config.limit_ladder)
                position.ladder_fill_prices = [0] * len(bot_config.limit_ladder)
                for tier in bot_config.limit_ladder:
                    tier_qty = max(1, int(tier.stake_dollars * 100) // max(1, tier.price_cents))
                    tier_expiry_ts = int(datetime.now(timezone.utc).timestamp() + tier.window_minutes * 60)
                    if self.is_live_trading:
                        tier_result = await self._execute_kalshi_order(
                            ticker=market_ticker,
                            side=side.value,
                            action="buy",
                            count=tier_qty,
                            price_cents=tier.price_cents,
                            order_type="limit",
                            expiration_ts=tier_expiry_ts
                        )
                        tier_order_id = tier_result.get("order_id", "") if tier_result and "error" not in tier_result else ""
                        if tier_order_id:
                            logger.info(f"Ladder tier {tier.price_cents}¢ order placed: {tier_order_id}")
                        else:
                            logger.warning(f"Ladder tier {tier.price_cents}¢ order failed: {tier_result}")
                    else:
                        tier_order_id = f"paper-ladder-{tier.price_cents}"
                    position.ladder_order_ids.append(tier_order_id)

            self.positions[position.id] = position

            # Save to database
            await self._save_position_to_db(position)

            # Determine order type description
            order_type_desc = f"LIMIT @ {position.entry_price_cents}¢" if should_use_limit else f"MARKET @ {position.entry_price_cents}¢"
            logger.info(f"Opened position: {side.value.upper()} {quantity}x {market_ticker} - {order_type_desc} (live={self.is_live_trading})")

            # Log position opened with clear side indication
            # YES = betting price goes UP (bullish)
            # NO = betting price goes DOWN (bearish)
            side_emoji = "🟢" if side == PositionSide.YES else "🔴"
            side_desc = "YES (bullish)" if side == PositionSide.YES else "NO (bearish)"

            self._add_log(
                level=LogLevel.TRADE,
                action="position_opened",
                message=f"{side_emoji} Opened {side_desc}: {quantity}x @ {position.entry_price_cents}¢ {order_type_desc}",
                bot_id=bot_id,
                bot_name=bot.name,
                details={
                    "position_id": position.id,
                    "market_ticker": market_ticker,
                    "side": side.value,
                    "side_description": "Betting UP" if side == PositionSide.YES else "Betting DOWN",
                    "quantity": quantity,
                    "entry_price_cents": position.entry_price_cents,
                    "order_type": "limit" if should_use_limit else "market",
                    "order_status": position.order_status.value,
                    "kalshi_order_id": position.kalshi_order_id,
                    "live_trading": self.is_live_trading
                }
            )
            return position

    async def _execute_kalshi_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price_cents: Optional[int] = None,
        order_type: str = "market",
        expiration_ts: Optional[int] = None
    ) -> Optional[Dict]:
        """Execute an order on Kalshi."""
        if not self._kalshi_client:
            return None

        try:
            logger.info(f"Executing {order_type} order: {action} {count}x {side} @ {price_cents}¢ on {ticker}")

            # Validate inputs
            if count <= 0:
                logger.error(f"Invalid count: {count} (must be > 0)")
                return {"error": f"Invalid count: {count}", "status_code": 400}

            if order_type == "limit" and (price_cents is None or price_cents < 1 or price_cents > 99):
                logger.error(f"Invalid limit price: {price_cents}¢ (must be 1-99)")
                return {"error": f"Invalid limit price: {price_cents}", "status_code": 400}

            # For limit orders, send the appropriate price based on side
            # Kalshi requires yes_price for yes side, no_price for no side
            if order_type == "limit":
                yes_price = price_cents if side == "yes" else None
                no_price = price_cents if side == "no" else None
            else:
                # Market orders - no price needed
                yes_price = None
                no_price = None

            logger.info(f"Order params: ticker={ticker}, side={side}, action={action}, count={count}, type={order_type}, yes_price={yes_price}, no_price={no_price}")

            result = await self._kalshi_client.place_order(
                ticker=ticker,
                side=side,
                action=action,
                count=count,
                type=order_type,
                yes_price=yes_price,
                no_price=no_price,
                expiration_ts=expiration_ts
            )
            return result
        except Exception as e:
            logger.error(f"Kalshi order execution error: {e}")
            return {"error": str(e)}

    async def _check_risk_limits(self, bot: Bot, cost_cents: int) -> bool:
        """
        Check if opening a position would violate risk limits.

        Returns True if position is allowed, False if blocked.
        """
        config = bot.config

        # Check daily loss limit
        if config.daily_loss_limit_cents > 0:
            if bot.daily_pnl_cents < -config.daily_loss_limit_cents:
                logger.warning(f"Daily loss limit reached: {bot.daily_pnl_cents}¢")
                return False

        # Check daily trade limit
        if config.daily_trade_limit > 0:
            if bot.daily_trades >= config.daily_trade_limit:
                logger.warning(f"Daily trade limit reached: {bot.daily_trades}")
                return False

        # max_position_size caps contracts *per trade* (enforced in _calculate_stake).
        # No concurrent-position block here — each Kalshi candle is a different market,
        # so a position in candle N must not prevent trading candle N+1.

        return True

    async def close_position(self, position_id: str, exit_price_cents: int, use_limit_order: bool = False) -> Optional[Trade]:
        """
        Close a position and record the trade.

        Args:
            position_id: ID of position to close
            exit_price_cents: Exit price in cents
            use_limit_order: If True, use limit order at exit_price_cents

        Returns:
            Trade record if successful
        """
        async with self._lock:
            position = self.positions.get(position_id)
            if not position:
                return None

            # Execute close order on Kalshi if live trading
            if self.is_live_trading and position.kalshi_order_id:
                close_result = await self._execute_kalshi_order(
                    ticker=position.market_ticker,
                    side=position.side.value,
                    action="sell",
                    count=position.filled_quantity or position.quantity,
                    price_cents=exit_price_cents if use_limit_order else None,
                    order_type="limit" if use_limit_order else "market"
                )

                if close_result and "error" not in close_result:
                    # Update exit price if filled at different price
                    if close_result.get("yes_price"):
                        exit_price_cents = close_result.get("yes_price")
                    logger.info(f"Kalshi close order executed: {close_result.get('order_id')}")
                else:
                    error_msg = close_result.get("error", "Unknown") if close_result else "No response"
                    logger.error(f"Failed to close on Kalshi: {error_msg}")
                    # Continue with local tracking anyway

            # Calculate PnL
            if position.side == PositionSide.YES:
                pnl_cents = (exit_price_cents - position.entry_price_cents) * position.quantity
            else:
                pnl_cents = (position.entry_price_cents - exit_price_cents) * position.quantity

            # Determine result
            if pnl_cents > 0:
                result = "win"
            elif pnl_cents < 0:
                result = "loss"
            else:
                result = "breakeven"

            # Create trade record
            trade = Trade(
                bot_id=position.bot_id,
                market_ticker=position.market_ticker,
                market_title=position.market_title,
                side=position.side,
                entry_price_cents=position.entry_price_cents,
                exit_price_cents=exit_price_cents,
                quantity=position.quantity,
                entry_time=position.entry_time,
                exit_time=datetime.now(timezone.utc),
                pnl_cents=pnl_cents,
                result=result
            )

            self.trades.append(trade)

            # Save trade to database
            await self._save_trade_to_db(trade)

            # Update bot metrics
            await self._update_metrics(position.bot_id, trade)

            # Update bot state
            bot = self.bots.get(position.bot_id)
            if bot:
                bot.last_trade_at = trade.exit_time
                bot.daily_pnl_cents += pnl_cents
                bot.daily_trades += 1

                # Update martingale level
                if bot.config.use_martingale:
                    if result == "loss":
                        bot.current_martingale_level = min(
                            bot.current_martingale_level + 1,
                            bot.config.max_martingale_level
                        )
                    else:
                        bot.current_martingale_level = 0

            # Remove position
            del self.positions[position_id]

            # Delete position from database
            if self._db:
                await self._db.delete_position(position_id)

            # Save updated bot and metrics to database
            if bot:
                await self._save_bot_to_db(bot)
            await self._save_metrics_to_db(position.bot_id)

            logger.info(f"Closed position: {result} {pnl_cents}¢ (live={self.is_live_trading})")

            # Log position closed
            result_emoji = {"win": "✅", "loss": "❌", "breakeven": "➖"}
            pnl_sign = "+" if pnl_cents >= 0 else ""

            self._add_log(
                level=LogLevel.TRADE,
                action="position_closed",
                message=f"{result_emoji.get(result, '📊')} Closed position: {result.upper()} {pnl_sign}{pnl_cents}¢",
                bot_id=position.bot_id,
                bot_name=bot.name if bot else None,
                details={
                    "trade_id": trade.id,
                    "position_id": position_id,
                    "market_ticker": position.market_ticker,
                    "side": position.side.value,
                    "entry_price_cents": position.entry_price_cents,
                    "exit_price_cents": exit_price_cents,
                    "quantity": position.quantity,
                    "pnl_cents": pnl_cents,
                    "result": result
                }
            )
            return trade

    async def get_positions(self, bot_id: Optional[str] = None) -> List[Position]:
        """Get open positions, optionally filtered by bot."""
        positions = list(self.positions.values())
        if bot_id:
            positions = [p for p in positions if p.bot_id == bot_id]
        return positions

    async def update_position_prices(self, market_ticker: str, current_price_cents: int):
        """Update current prices for all positions in a market."""
        for position in self.positions.values():
            if position.market_ticker == market_ticker:
                position.update_pnl(current_price_cents)

    # =====================
    # Metrics & History
    # =====================

    async def _update_metrics(self, bot_id: str, trade: Trade):
        """Update bot metrics after a trade."""
        metrics = self.metrics.get(bot_id)
        if not metrics:
            metrics = BotMetrics()
            self.metrics[bot_id] = metrics

        metrics.total_trades += 1
        metrics.total_pnl_cents += trade.pnl_cents
        metrics.total_pnl_dollars = metrics.total_pnl_cents / 100.0

        if trade.result == "win":
            metrics.winning_trades += 1
            metrics.largest_win_cents = max(metrics.largest_win_cents, trade.pnl_cents)
            if metrics.current_streak >= 0:
                metrics.current_streak += 1
            else:
                metrics.current_streak = 1
            metrics.best_streak = max(metrics.best_streak, metrics.current_streak)

        elif trade.result == "loss":
            metrics.losing_trades += 1
            metrics.largest_loss_cents = min(metrics.largest_loss_cents, trade.pnl_cents)
            if metrics.current_streak <= 0:
                metrics.current_streak -= 1
            else:
                metrics.current_streak = -1
            metrics.worst_streak = min(metrics.worst_streak, metrics.current_streak)

        else:
            metrics.breakeven_trades += 1

        # Calculate win rate
        if metrics.total_trades > 0:
            metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100

        # Calculate averages
        if metrics.winning_trades > 0:
            total_wins = sum(t.pnl_cents for t in self.trades if t.bot_id == bot_id and t.result == "win")
            metrics.avg_win_cents = total_wins / metrics.winning_trades

        if metrics.losing_trades > 0:
            total_losses = sum(t.pnl_cents for t in self.trades if t.bot_id == bot_id and t.result == "loss")
            metrics.avg_loss_cents = total_losses / metrics.losing_trades

    async def get_metrics(self, bot_id: str) -> Optional[BotMetrics]:
        """Get metrics for a bot."""
        return self.metrics.get(bot_id)

    async def get_trades(self, bot_id: Optional[str] = None, limit: int = 50) -> List[Trade]:
        """Get trade history, optionally filtered by bot."""
        trades = self.trades
        if bot_id:
            trades = [t for t in trades if t.bot_id == bot_id]
        # Return most recent first
        return sorted(trades, key=lambda t: t.exit_time, reverse=True)[:limit]

    async def get_summary(self) -> dict:
        """Get overall summary across all bots."""
        total_bots = len(self.bots)
        running_bots = len([b for b in self.bots.values() if b.status == BotStatus.RUNNING])
        total_positions = len(self.positions)
        total_trades = len(self.trades)

        total_pnl = sum(t.pnl_cents for t in self.trades)
        total_unrealized = sum(p.unrealized_pnl_cents for p in self.positions.values())

        return {
            "total_bots": total_bots,
            "running_bots": running_bots,
            "total_positions": total_positions,
            "total_trades": total_trades,
            "total_realized_pnl_cents": total_pnl,
            "total_realized_pnl_dollars": total_pnl / 100.0,
            "total_unrealized_pnl_cents": total_unrealized,
            "total_unrealized_pnl_dollars": total_unrealized / 100.0,
            "live_trading_enabled": self.is_live_trading,
            "total_logs": len(self.logs)
        }

    def log_signal(self, direction: str, confidence: str, score_up: float, score_dn: float, price: float):
        """Log a signal computation."""
        direction_emoji = {"UP": "🟢", "DOWN": "🔴", "SKIP": "⏭️"}

        # Add action hint based on direction
        if direction == "UP":
            action_hint = "→ Buy YES (bullish)"
        elif direction == "DOWN":
            action_hint = "→ Buy NO (bearish)"
        else:
            action_hint = "→ No trade"

        self._add_log(
            level=LogLevel.SIGNAL,
            action="signal_computed",
            message=f"{direction_emoji.get(direction, '📊')} Signal: {direction} ({confidence}) @ ${price:,.2f} {action_hint}",
            details={
                "direction": direction,
                "confidence": confidence,
                "score_up": score_up,
                "score_dn": score_dn,
                "price": price,
                "recommended_side": "YES" if direction == "UP" else ("NO" if direction == "DOWN" else "NONE")
            }
        )

    # =====================
    # Auto-Trading Hooks
    # =====================

    async def on_signal(
        self,
        signal: "SignalResult",
        market_ticker: str,
        market_title: str,
        current_price_cents: int
    ) -> List[Position]:
        """
        Process a new signal and open positions for all eligible running bots.

        Called by the scheduler when a new 15-minute candle closes.

        Args:
            signal: Full SignalResult object with all indicator data
            market_ticker: Current Kalshi market ticker
            market_title: Human-readable market title
            current_price_cents: Current contract price in cents

        Returns:
            List of opened positions
        """
        opened_positions = []

        for bot in self.bots.values():
            if bot.status != BotStatus.RUNNING:
                continue

            # Determine trade direction based on bot's strategy mode
            trade_direction = self._get_trade_direction(bot, signal)

            if trade_direction == "SKIP":
                logger.debug(f"Bot {bot.name}: No trade signal")
                continue

            # Determine position side based on trade direction:
            # - UP signal → buy YES (betting BTC goes UP in next candle)
            # - DOWN signal → buy NO (betting BTC goes DOWN in next candle)
            side = PositionSide.YES if trade_direction == "UP" else PositionSide.NO

            # Use bot's configured limit price if limit orders enabled
            entry_price = bot.config.limit_price_cents if bot.config.use_limit_orders else current_price_cents

            # Calculate stake (contracts) — pass market price so dollar→contract
            # conversion uses the actual entry price, not a hardcoded fallback
            stake = self._calculate_stake(bot, market_price_cents=current_price_cents)

            position = await self.open_position(
                bot_id=bot.id,
                market_ticker=market_ticker,
                market_title=market_title,
                side=side,
                entry_price_cents=entry_price,
                quantity=stake
            )

            if position:
                opened_positions.append(position)
                side_desc = "YES (bullish)" if side == PositionSide.YES else "NO (bearish)"
                order_desc = f"LIMIT @ {entry_price}¢" if bot.config.use_limit_orders else f"MARKET @ {current_price_cents}¢"
                strategy_desc = f"[{bot.config.strategy_mode}]"
                logger.info(f"Auto-trade for {bot.name} {strategy_desc}: {trade_direction} → {side_desc} {stake}x {order_desc}")

        return opened_positions

    def _get_trade_direction(self, bot: Bot, signal: "SignalResult") -> str:
        """
        Determine trade direction based on bot's strategy mode and config.

        Returns: "UP", "DOWN", or "SKIP"
        """
        config = bot.config

        # Strategy: Follow the candle that just closed (signal_candle_color = df.iloc[-2])
        if config.strategy_mode == "follow_candle":
            if signal.signal_candle_color == "GREEN":
                logger.debug(f"Bot {bot.name}: Signal candle GREEN → UP")
                return "UP"
            elif signal.signal_candle_color == "RED":
                logger.debug(f"Bot {bot.name}: Signal candle RED → DOWN")
                return "DOWN"
            else:
                logger.debug(f"Bot {bot.name}: Signal candle NEUTRAL (doji) → SKIP")
                return "SKIP"

        # Strategy: Use 4-filter scoring system (default)
        # Check individual filter requirements based on bot config
        return self._evaluate_signal_filters(bot, signal)

    def _evaluate_signal_filters(self, bot: Bot, signal: "SignalResult") -> str:
        """
        Evaluate signal against bot's configured filter requirements.

        Returns: "UP", "DOWN", or "SKIP"
        """
        config = bot.config

        # Count passing filters for each direction
        score_up = 0.0
        score_dn = 0.0
        required_filters_met = True

        # EMA Alignment check
        if config.require_ema_alignment:
            if signal.ema_signal == "BULL":
                score_up += 1
            elif signal.ema_signal == "BEAR":
                score_dn += 1
            else:
                # EMA is required but neutral - don't trade
                required_filters_met = False
        else:
            # Not required, but still count for scoring
            if signal.ema_signal == "BULL":
                score_up += 1
            elif signal.ema_signal == "BEAR":
                score_dn += 1

        # RSI Zone check
        if config.require_rsi_zone:
            if signal.rsi_filter == "BULL":
                score_up += 1
            elif signal.rsi_filter == "BEAR":
                score_dn += 1
            else:
                # RSI is required but neutral - don't trade
                required_filters_met = False
        else:
            # Not required, but still count for scoring
            if signal.rsi_filter == "BULL":
                score_up += 1
            elif signal.rsi_filter == "BEAR":
                score_dn += 1

        # MACD Confirmation check
        if config.require_macd_confirmation:
            if signal.macd_filter == "BULL":
                score_up += 1
            elif signal.macd_filter == "BEAR":
                score_dn += 1
            # MACD is always BULL or BEAR, never neutral
        else:
            # Not required, but still count for scoring
            if signal.macd_filter == "BULL":
                score_up += 1
            elif signal.macd_filter == "BEAR":
                score_dn += 1

        # Close Position check
        if config.require_close_position:
            if signal.close_pos_filter == "BULL":
                score_up += 1
            elif signal.close_pos_filter == "BEAR":
                score_dn += 1
            else:
                # Close position is required but neutral - don't trade
                required_filters_met = False
        else:
            # Not required, but still count for scoring
            if signal.close_pos_filter == "BULL":
                score_up += 1
            elif signal.close_pos_filter == "BEAR":
                score_dn += 1

        # If required filters weren't met, skip
        if not required_filters_met:
            logger.debug(f"Bot {bot.name}: Required filters not met → SKIP")
            return "SKIP"

        # Determine direction based on score and min_score threshold
        if score_up >= config.min_score and score_up > score_dn:
            logger.debug(f"Bot {bot.name}: Score UP={score_up} >= {config.min_score} → UP")
            return "UP"
        elif score_dn >= config.min_score and score_dn > score_up:
            logger.debug(f"Bot {bot.name}: Score DN={score_dn} >= {config.min_score} → DOWN")
            return "DOWN"
        else:
            logger.debug(f"Bot {bot.name}: Score UP={score_up}, DN={score_dn}, min={config.min_score} → SKIP")
            return "SKIP"

    def _calculate_stake(self, bot: Bot, market_price_cents: int = 50) -> int:
        """
        Calculate stake in contracts, scaling the DOLLAR RISK by the martingale level.

        Martingale doubles the dollar amount risked per trade (not the contract count).
        Example with base_stake=$1.00, multiplier=2.0, limit price=45¢:
          Level 0: $1.00  → 100/45 = 2 contracts
          Level 1: $2.00  → 200/45 = 4 contracts
          Level 2: $4.00  → 400/45 = 8 contracts
          Level 3: $8.00  → 800/45 = 17 contracts
        max_position_size caps non-martingale trades; martingale is capped by
        max_martingale_level (the natural growth ceiling).
        """
        config = bot.config

        # Effective entry price per contract
        if config.use_limit_orders and config.limit_price_cents > 0:
            price = config.limit_price_cents
        else:
            price = max(1, market_price_cents)

        if config.use_martingale and bot.current_martingale_level > 0:
            # Double the DOLLAR stake at each level, then convert to contracts
            multiplier = config.martingale_multiplier ** bot.current_martingale_level
            stake_cents = int(config.base_stake_cents * multiplier)
            contracts = max(1, stake_cents // price)
            logger.info(
                f"Martingale level {bot.current_martingale_level}: "
                f"${stake_cents/100:.2f} stake → {contracts} contracts @ {price}¢"
            )
            return contracts

        # Base case: convert base dollar stake to contracts — matches backtest sizing exactly
        return max(1, config.base_stake_cents // price)

    async def sync_kalshi_positions(self) -> int:
        """
        Sync local positions with Kalshi API.

        Returns:
            Number of positions synced
        """
        if not self._kalshi_client:
            return 0

        try:
            kalshi_positions = await self._kalshi_client.get_positions()
            synced = 0

            for kp in kalshi_positions:
                ticker = kp.get("ticker")
                position_count = kp.get("position", 0)

                if position_count == 0:
                    continue

                # Check if we have this position locally
                local_match = None
                for pos in self.positions.values():
                    if pos.market_ticker == ticker:
                        local_match = pos
                        break

                if local_match:
                    # Update local position
                    local_match.filled_quantity = abs(position_count)
                    local_match.order_status = OrderStatus.FILLED
                    synced += 1
                else:
                    logger.warning(f"Found untracked Kalshi position: {ticker} ({position_count})")

            logger.info(f"Synced {synced} positions with Kalshi")
            return synced

        except Exception as e:
            logger.error(f"Error syncing Kalshi positions: {e}")
            return 0

    async def get_kalshi_balance(self) -> Optional[Dict]:
        """Get current Kalshi account balance."""
        if not self._kalshi_client:
            return None

        return await self._kalshi_client.get_balance()

    async def reset_daily_stats(self):
        """Reset daily stats for all bots. Call at midnight UTC."""
        async with self._lock:
            for bot in self.bots.values():
                bot.daily_pnl_cents = 0
                bot.daily_trades = 0
                await self._save_bot_to_db(bot)
            if self._db:
                await self._db.reset_daily_stats()
            logger.info("Daily stats reset for all bots")

    async def sync_order_status(self, position_id: str = None) -> Dict[str, Any]:
        """
        Sync order status from Kalshi API for live trading positions.

        Args:
            position_id: Specific position to sync, or None for all positions

        Returns:
            Dict with sync results
        """
        if not self._kalshi_client or not self._live_trading_enabled:
            return {"synced": 0, "message": "Live trading not enabled"}

        results = {"synced": 0, "updated": [], "errors": []}

        positions_to_check = []
        if position_id:
            pos = self.positions.get(position_id)
            if pos and pos.kalshi_order_id:
                positions_to_check.append(pos)
        else:
            positions_to_check = [
                p for p in self.positions.values()
                if p.kalshi_order_id and p.order_status == OrderStatus.PENDING
            ]

        for position in positions_to_check:
            try:
                # Fetch order status from Kalshi
                order = await self._kalshi_client.get_order(position.kalshi_order_id)
                if not order:
                    continue

                old_status = position.order_status
                kalshi_status = order.get("status", "").lower()

                # Map Kalshi status to our OrderStatus
                new_status = old_status
                if kalshi_status == "executed":
                    new_status = OrderStatus.FILLED
                    position.filled_quantity = order.get("count_executed", position.quantity)
                elif kalshi_status == "resting":
                    new_status = OrderStatus.PENDING
                elif kalshi_status == "canceled" or kalshi_status == "cancelled":
                    new_status = OrderStatus.CANCELLED
                elif kalshi_status == "partial":
                    new_status = OrderStatus.PARTIALLY_FILLED
                    position.filled_quantity = order.get("count_executed", 0)

                if new_status != old_status:
                    position.order_status = new_status
                    await self._save_position_to_db(position)

                    # Log the status change
                    status_emoji = {
                        OrderStatus.FILLED: "✅",
                        OrderStatus.CANCELLED: "❌",
                        OrderStatus.PARTIALLY_FILLED: "🔶"
                    }

                    bot = self.bots.get(position.bot_id)
                    self._add_log(
                        level=LogLevel.TRADE if new_status == OrderStatus.FILLED else LogLevel.INFO,
                        action="order_status_changed",
                        message=f"{status_emoji.get(new_status, '📊')} Order {new_status.value}: {position.side.value.upper()} {position.quantity}x @ {position.entry_price_cents}¢",
                        bot_id=position.bot_id,
                        bot_name=bot.name if bot else None,
                        details={
                            "position_id": position.id,
                            "kalshi_order_id": position.kalshi_order_id,
                            "old_status": old_status.value,
                            "new_status": new_status.value,
                            "filled_quantity": position.filled_quantity
                        }
                    )

                    results["updated"].append({
                        "position_id": position.id,
                        "old_status": old_status.value,
                        "new_status": new_status.value
                    })

                results["synced"] += 1

            except Exception as e:
                logger.error(f"Error syncing order {position.kalshi_order_id}: {e}")
                results["errors"].append({
                    "position_id": position.id,
                    "error": str(e)
                })

        if results["updated"]:
            logger.info(f"Synced {results['synced']} orders, {len(results['updated'])} status changes")

        return results

    async def check_settlements(self) -> int:
        """
        Check if any open positions are in markets that have settled.
        Auto-closes positions at settlement price (100¢ for YES win, 0¢ for YES loss).

        Returns:
            Number of positions settled
        """
        if not self._kalshi_client:
            return 0

        settled_count = 0
        positions_to_settle = []

        # Collect filled positions that need settlement checking
        for position in list(self.positions.values()):
            if position.order_status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                positions_to_settle.append(position)

        for position in positions_to_settle:
            try:
                # Ask Kalshi API for the market's current status
                market = await self._kalshi_client.get_market(position.market_ticker)
                if not market:
                    continue

                market_status = market.get("status", "")
                market_result = market.get("result", "")  # "yes" or "no"

                if market_status not in ("settled", "finalized"):
                    continue

                # Market has settled — determine exit price
                if market_result == "yes":
                    exit_price = 100  # YES contracts pay out 100¢
                elif market_result == "no":
                    exit_price = 0  # YES contracts worth 0¢
                else:
                    logger.warning(f"Unknown market result '{market_result}' for {position.market_ticker}")
                    continue

                logger.info(
                    f"Market {position.market_ticker} settled: result={market_result}, "
                    f"closing {position.side.value.upper()} position @ {exit_price}¢"
                )

                # Close the position at settlement price (don't send Kalshi order — it's already settled)
                # Calculate PnL
                qty = position.filled_quantity or position.quantity
                if position.side == PositionSide.YES:
                    pnl_cents = (exit_price - position.entry_price_cents) * qty
                else:
                    pnl_cents = (position.entry_price_cents - exit_price) * qty

                result = "win" if pnl_cents > 0 else ("loss" if pnl_cents < 0 else "breakeven")

                trade = Trade(
                    bot_id=position.bot_id,
                    market_ticker=position.market_ticker,
                    market_title=position.market_title,
                    side=position.side,
                    entry_price_cents=position.entry_price_cents,
                    exit_price_cents=exit_price,
                    quantity=qty,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(timezone.utc),
                    pnl_cents=pnl_cents,
                    result=result
                )

                self.trades.append(trade)
                await self._save_trade_to_db(trade)
                await self._update_metrics(position.bot_id, trade)

                # Update bot state
                bot = self.bots.get(position.bot_id)
                if bot:
                    bot.last_trade_at = trade.exit_time
                    bot.daily_pnl_cents += pnl_cents
                    bot.daily_trades += 1

                    if bot.config.use_martingale:
                        if result == "loss":
                            bot.current_martingale_level = min(
                                bot.current_martingale_level + 1,
                                bot.config.max_martingale_level
                            )
                        else:
                            bot.current_martingale_level = 0

                # Remove position
                del self.positions[position.id]
                if self._db:
                    await self._db.delete_position(position.id)

                if bot:
                    await self._save_bot_to_db(bot)
                await self._save_metrics_to_db(position.bot_id)

                # Log it
                result_emoji = {"win": "✅", "loss": "❌", "breakeven": "➖"}
                pnl_sign = "+" if pnl_cents >= 0 else ""
                self._add_log(
                    level=LogLevel.TRADE,
                    action="position_settled",
                    message=f"{result_emoji.get(result, '📊')} Market settled ({market_result.upper()}): {position.side.value.upper()} {qty}x | Entry: {position.entry_price_cents}¢ → Settlement: {exit_price}¢ | PnL: {pnl_sign}{pnl_cents}¢",
                    bot_id=position.bot_id,
                    bot_name=bot.name if bot else None,
                    details={
                        "position_id": position.id,
                        "market_ticker": position.market_ticker,
                        "market_result": market_result,
                        "entry_price": position.entry_price_cents,
                        "exit_price": exit_price,
                        "pnl_cents": pnl_cents,
                        "result": result
                    }
                )

                settled_count += 1
                logger.info(f"Auto-settled position {position.id}: {result} {pnl_sign}{pnl_cents}¢")

            except Exception as e:
                logger.error(f"Error checking settlement for {position.market_ticker}: {e}")

        if settled_count:
            logger.info(f"Auto-settled {settled_count} position(s)")

        return settled_count

    async def cleanup_expired_positions(self) -> int:
        """
        Remove positions with expired limit orders that never filled.

        Returns:
            Number of positions cleaned up
        """
        now = datetime.now(timezone.utc)
        expired_positions = []

        async with self._lock:
            for pos_id, position in list(self.positions.items()):
                # Only clean up pending limit orders that have expired
                if (position.is_limit_order and
                    position.order_status == OrderStatus.PENDING and
                    position.limit_order_expiry and
                    now > position.limit_order_expiry):
                    expired_positions.append(position)

            for position in expired_positions:
                # Log the expiry
                bot = self.bots.get(position.bot_id)
                self._add_log(
                    level=LogLevel.WARNING,
                    action="order_expired",
                    message=f"⏰ Limit order expired (unfilled): {position.side.value.upper()} {position.quantity}x @ {position.entry_price_cents}¢",
                    bot_id=position.bot_id,
                    bot_name=bot.name if bot else None,
                    details={
                        "position_id": position.id,
                        "market_ticker": position.market_ticker,
                        "side": position.side.value,
                        "entry_price_cents": position.entry_price_cents,
                        "quantity": position.quantity,
                        "expiry_time": position.limit_order_expiry.isoformat() if position.limit_order_expiry else None
                    }
                )

                # Remove from memory
                del self.positions[position.id]

                # Remove from database
                if self._db:
                    await self._db.delete_position(position.id)

                logger.info(f"Cleaned up expired position: {position.id}")

        if expired_positions:
            logger.info(f"Cleaned up {len(expired_positions)} expired position(s)")

        return len(expired_positions)

    async def start_cleanup_task(self):
        """Start background task to periodically clean up expired positions and sync order status."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(30)  # Check every 30 seconds
                    await self.cleanup_expired_positions()
                    # Also sync order status and check settlements from Kalshi if live trading
                    if self._live_trading_enabled and self._kalshi_client:
                        await self.sync_order_status()
                        await self.check_settlements()
                    # Paper trading: check settlements too (market data still accessible)
                    elif self._kalshi_client:
                        await self.check_settlements()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Position cleanup task started (includes order status sync)")

    async def stop_cleanup_task(self):
        """Stop the cleanup background task."""
        if hasattr(self, '_cleanup_task') and self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Position cleanup task stopped")


# Global instance
bot_manager = BotManager()
