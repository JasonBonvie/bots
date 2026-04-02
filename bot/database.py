"""
SQLite database persistence layer for bots, positions, and trades.

Uses SQLAlchemy with async support for non-blocking database operations.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey,
    create_engine, event
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.future import select

import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

# Database file path
DB_PATH = Path(__file__).parent.parent / "data" / "nekalshi.db"


# =====================
# SQLAlchemy Models
# =====================

class BotModel(Base):
    """Persistent bot configuration and state."""
    __tablename__ = "bots"

    id = Column(String(8), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, default="")
    status = Column(String(20), default="stopped")  # running, paused, stopped
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_trade_at = Column(DateTime, nullable=True)

    # Daily tracking
    daily_pnl_cents = Column(Integer, default=0)
    daily_trades = Column(Integer, default=0)
    current_martingale_level = Column(Integer, default=0)

    # Config stored as JSON
    config_json = Column(Text, default="{}")

    # Relationships
    positions = relationship("PositionModel", back_populates="bot", cascade="all, delete-orphan")
    trades = relationship("TradeModel", back_populates="bot", cascade="all, delete-orphan")

    def get_config(self) -> Dict[str, Any]:
        """Parse config JSON."""
        try:
            return json.loads(self.config_json or "{}")
        except json.JSONDecodeError:
            return {}

    def set_config(self, config: Dict[str, Any]):
        """Serialize config to JSON."""
        self.config_json = json.dumps(config)


class PositionModel(Base):
    """Persistent open positions."""
    __tablename__ = "positions"

    id = Column(String(8), primary_key=True)
    bot_id = Column(String(8), ForeignKey("bots.id"), nullable=False)
    market_ticker = Column(String(100), nullable=False)
    market_title = Column(String(200), default="")
    side = Column(String(10), nullable=False)  # yes, no
    entry_price_cents = Column(Integer, nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    current_price_cents = Column(Integer, default=0)
    unrealized_pnl_cents = Column(Integer, default=0)

    # Kalshi order tracking
    kalshi_order_id = Column(String(50), nullable=True)
    order_status = Column(String(20), default="pending")  # pending, filled, partial, cancelled, rejected
    filled_quantity = Column(Integer, default=0)
    rejection_reason = Column(Text, nullable=True)  # Why order was rejected (if applicable)

    # Limit order expiry tracking
    limit_order_expiry = Column(DateTime, nullable=True)  # When the limit order expires
    is_limit_order = Column(Boolean, default=False)

    # Relationship
    bot = relationship("BotModel", back_populates="positions")


class TradeModel(Base):
    """Persistent trade history."""
    __tablename__ = "trades"

    id = Column(String(8), primary_key=True)
    bot_id = Column(String(8), ForeignKey("bots.id"), nullable=False)
    market_ticker = Column(String(100), nullable=False)
    market_title = Column(String(200), default="")
    side = Column(String(10), nullable=False)  # yes, no
    entry_price_cents = Column(Integer, nullable=False)
    exit_price_cents = Column(Integer, nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    pnl_cents = Column(Integer, nullable=False)
    result = Column(String(20), nullable=False)  # win, loss, breakeven

    # Relationship
    bot = relationship("BotModel", back_populates="trades")


class BotMetricsModel(Base):
    """Persistent bot performance metrics."""
    __tablename__ = "bot_metrics"

    bot_id = Column(String(8), ForeignKey("bots.id"), primary_key=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    breakeven_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_pnl_cents = Column(Integer, default=0)
    avg_win_cents = Column(Float, default=0.0)
    avg_loss_cents = Column(Float, default=0.0)
    largest_win_cents = Column(Integer, default=0)
    largest_loss_cents = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)
    best_streak = Column(Integer, default=0)
    worst_streak = Column(Integer, default=0)


class LogModel(Base):
    """Persistent activity logs."""
    __tablename__ = "logs"

    id = Column(String(8), primary_key=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    bot_id = Column(String(8), nullable=True, index=True)
    bot_name = Column(String(100), nullable=True)
    level = Column(String(20), default="info")  # info, success, warning, error, trade, signal
    action = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    details_json = Column(Text, nullable=True)

    def get_details(self) -> Optional[Dict[str, Any]]:
        """Parse details JSON."""
        if not self.details_json:
            return None
        try:
            return json.loads(self.details_json)
        except json.JSONDecodeError:
            return None

    def set_details(self, details: Optional[Dict[str, Any]]):
        """Serialize details to JSON."""
        if details:
            self.details_json = json.dumps(details)
        else:
            self.details_json = None


# =====================
# Database Manager
# =====================

class Database:
    """Async database manager for SQLite persistence."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.engine = None
        self.async_session = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection and create tables."""
        if self._initialized:
            return

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create async engine
        db_url = f"sqlite+aiosqlite:///{self.db_path}"
        self.engine = create_async_engine(db_url, echo=False)

        # Create session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._initialized = True
        logger.info(f"Database initialized at {self.db_path}")

    async def close(self):
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False

    def session(self) -> AsyncSession:
        """Get a new database session."""
        if not self.async_session:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.async_session()

    # =====================
    # Bot Operations
    # =====================

    async def save_bot(self, bot_data: Dict[str, Any]) -> BotModel:
        """Save or update a bot."""
        async with self.session() as session:
            # Check if bot exists
            result = await session.execute(
                select(BotModel).where(BotModel.id == bot_data["id"])
            )
            bot = result.scalar_one_or_none()

            if bot:
                # Update existing bot
                bot.name = bot_data.get("name", bot.name)
                bot.description = bot_data.get("description", bot.description)
                bot.status = bot_data.get("status", bot.status)
                bot.updated_at = datetime.now(timezone.utc)
                bot.last_trade_at = bot_data.get("last_trade_at")
                bot.daily_pnl_cents = bot_data.get("daily_pnl_cents", 0)
                bot.daily_trades = bot_data.get("daily_trades", 0)
                bot.current_martingale_level = bot_data.get("current_martingale_level", 0)
                if "config" in bot_data:
                    bot.set_config(bot_data["config"])
            else:
                # Create new bot
                bot = BotModel(
                    id=bot_data["id"],
                    name=bot_data["name"],
                    description=bot_data.get("description", ""),
                    status=bot_data.get("status", "stopped"),
                    created_at=bot_data.get("created_at", datetime.now(timezone.utc)),
                    updated_at=datetime.now(timezone.utc),
                    daily_pnl_cents=bot_data.get("daily_pnl_cents", 0),
                    daily_trades=bot_data.get("daily_trades", 0),
                    current_martingale_level=bot_data.get("current_martingale_level", 0)
                )
                if "config" in bot_data:
                    bot.set_config(bot_data["config"])
                session.add(bot)

            await session.commit()
            return bot

    async def get_bot(self, bot_id: str) -> Optional[BotModel]:
        """Get a bot by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(BotModel).where(BotModel.id == bot_id)
            )
            return result.scalar_one_or_none()

    async def get_all_bots(self) -> List[BotModel]:
        """Get all bots."""
        async with self.session() as session:
            result = await session.execute(select(BotModel))
            return list(result.scalars().all())

    async def delete_bot(self, bot_id: str) -> bool:
        """Delete a bot and all related data."""
        async with self.session() as session:
            result = await session.execute(
                select(BotModel).where(BotModel.id == bot_id)
            )
            bot = result.scalar_one_or_none()
            if bot:
                await session.delete(bot)
                await session.commit()
                return True
            return False

    async def update_bot_status(self, bot_id: str, status: str) -> bool:
        """Update bot status."""
        async with self.session() as session:
            result = await session.execute(
                select(BotModel).where(BotModel.id == bot_id)
            )
            bot = result.scalar_one_or_none()
            if bot:
                bot.status = status
                bot.updated_at = datetime.now(timezone.utc)
                await session.commit()
                return True
            return False

    # =====================
    # Position Operations
    # =====================

    async def save_position(self, position_data: Dict[str, Any]) -> PositionModel:
        """Save or update a position."""
        async with self.session() as session:
            result = await session.execute(
                select(PositionModel).where(PositionModel.id == position_data["id"])
            )
            position = result.scalar_one_or_none()

            if position:
                # Update existing position
                position.current_price_cents = position_data.get("current_price_cents", position.current_price_cents)
                position.unrealized_pnl_cents = position_data.get("unrealized_pnl_cents", position.unrealized_pnl_cents)
                position.order_status = position_data.get("order_status", position.order_status)
                position.filled_quantity = position_data.get("filled_quantity", position.filled_quantity)
                position.kalshi_order_id = position_data.get("kalshi_order_id", position.kalshi_order_id)
                position.limit_order_expiry = position_data.get("limit_order_expiry", position.limit_order_expiry)
                position.is_limit_order = position_data.get("is_limit_order", position.is_limit_order)
                position.rejection_reason = position_data.get("rejection_reason", position.rejection_reason)
            else:
                # Create new position
                position = PositionModel(
                    id=position_data["id"],
                    bot_id=position_data["bot_id"],
                    market_ticker=position_data["market_ticker"],
                    market_title=position_data.get("market_title", ""),
                    side=position_data["side"],
                    entry_price_cents=position_data["entry_price_cents"],
                    quantity=position_data["quantity"],
                    entry_time=position_data.get("entry_time", datetime.now(timezone.utc)),
                    current_price_cents=position_data.get("current_price_cents", position_data["entry_price_cents"]),
                    unrealized_pnl_cents=position_data.get("unrealized_pnl_cents", 0),
                    kalshi_order_id=position_data.get("kalshi_order_id"),
                    order_status=position_data.get("order_status", "pending"),
                    filled_quantity=position_data.get("filled_quantity", 0),
                    limit_order_expiry=position_data.get("limit_order_expiry"),
                    is_limit_order=position_data.get("is_limit_order", False),
                    rejection_reason=position_data.get("rejection_reason")
                )
                session.add(position)

            await session.commit()
            return position

    async def get_position(self, position_id: str) -> Optional[PositionModel]:
        """Get a position by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(PositionModel).where(PositionModel.id == position_id)
            )
            return result.scalar_one_or_none()

    async def get_positions(self, bot_id: Optional[str] = None) -> List[PositionModel]:
        """Get positions, optionally filtered by bot."""
        async with self.session() as session:
            query = select(PositionModel)
            if bot_id:
                query = query.where(PositionModel.bot_id == bot_id)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def delete_position(self, position_id: str) -> bool:
        """Delete a position."""
        async with self.session() as session:
            result = await session.execute(
                select(PositionModel).where(PositionModel.id == position_id)
            )
            position = result.scalar_one_or_none()
            if position:
                await session.delete(position)
                await session.commit()
                return True
            return False

    # =====================
    # Trade Operations
    # =====================

    async def save_trade(self, trade_data: Dict[str, Any]) -> TradeModel:
        """Save a trade."""
        async with self.session() as session:
            trade = TradeModel(
                id=trade_data["id"],
                bot_id=trade_data["bot_id"],
                market_ticker=trade_data["market_ticker"],
                market_title=trade_data.get("market_title", ""),
                side=trade_data["side"],
                entry_price_cents=trade_data["entry_price_cents"],
                exit_price_cents=trade_data["exit_price_cents"],
                quantity=trade_data["quantity"],
                entry_time=trade_data["entry_time"],
                exit_time=trade_data.get("exit_time", datetime.now(timezone.utc)),
                pnl_cents=trade_data["pnl_cents"],
                result=trade_data["result"]
            )
            session.add(trade)
            await session.commit()
            return trade

    async def get_trades(self, bot_id: Optional[str] = None, limit: int = 50) -> List[TradeModel]:
        """Get trades, optionally filtered by bot."""
        async with self.session() as session:
            query = select(TradeModel).order_by(TradeModel.exit_time.desc()).limit(limit)
            if bot_id:
                query = query.where(TradeModel.bot_id == bot_id)
            result = await session.execute(query)
            return list(result.scalars().all())

    # =====================
    # Metrics Operations
    # =====================

    async def save_metrics(self, bot_id: str, metrics_data: Dict[str, Any]) -> BotMetricsModel:
        """Save or update bot metrics."""
        async with self.session() as session:
            result = await session.execute(
                select(BotMetricsModel).where(BotMetricsModel.bot_id == bot_id)
            )
            metrics = result.scalar_one_or_none()

            if metrics:
                # Update existing metrics
                for key, value in metrics_data.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)
            else:
                # Create new metrics
                metrics = BotMetricsModel(bot_id=bot_id, **metrics_data)
                session.add(metrics)

            await session.commit()
            return metrics

    async def get_metrics(self, bot_id: str) -> Optional[BotMetricsModel]:
        """Get bot metrics."""
        async with self.session() as session:
            result = await session.execute(
                select(BotMetricsModel).where(BotMetricsModel.bot_id == bot_id)
            )
            return result.scalar_one_or_none()

    # =====================
    # Log Operations
    # =====================

    async def save_log(self, log_data: Dict[str, Any]) -> LogModel:
        """Save a log entry."""
        async with self.session() as session:
            log = LogModel(
                id=log_data["id"],
                timestamp=log_data.get("timestamp", datetime.now(timezone.utc)),
                bot_id=log_data.get("bot_id"),
                bot_name=log_data.get("bot_name"),
                level=log_data.get("level", "info"),
                action=log_data["action"],
                message=log_data["message"]
            )
            if "details" in log_data:
                log.set_details(log_data["details"])
            session.add(log)
            await session.commit()
            return log

    async def get_logs(self, bot_id: Optional[str] = None, limit: int = 100) -> List[LogModel]:
        """Get logs, optionally filtered by bot."""
        async with self.session() as session:
            query = select(LogModel).order_by(LogModel.timestamp.desc()).limit(limit)
            if bot_id:
                query = query.where(LogModel.bot_id == bot_id)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def clear_old_logs(self, keep_count: int = 1000):
        """Clear old logs, keeping the most recent ones."""
        async with self.session() as session:
            # Get IDs of logs to keep
            subquery = select(LogModel.id).order_by(LogModel.timestamp.desc()).limit(keep_count)
            result = await session.execute(subquery)
            keep_ids = [row[0] for row in result.fetchall()]

            # Delete logs not in keep list
            if keep_ids:
                from sqlalchemy import delete
                await session.execute(
                    delete(LogModel).where(LogModel.id.notin_(keep_ids))
                )
                await session.commit()

    # =====================
    # Utility Operations
    # =====================

    async def reset_daily_stats(self):
        """Reset daily stats for all bots (call at midnight UTC)."""
        async with self.session() as session:
            result = await session.execute(select(BotModel))
            bots = result.scalars().all()
            for bot in bots:
                bot.daily_pnl_cents = 0
                bot.daily_trades = 0
            await session.commit()
            logger.info("Daily stats reset for all bots")


# Global database instance
db = Database()
