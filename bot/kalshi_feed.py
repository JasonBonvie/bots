"""
Kalshi API integration for live market data.
Provides real-time BTC market prices directly from Kalshi.

Kalshi API Documentation: https://trading-api.kalshi.com/docs
Uses RSA signing for authentication.
"""

import asyncio
import json
import logging
import time
import base64
from datetime import datetime, timezone
from typing import Set, Optional, Dict, List
from pydantic import BaseModel
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# RSA signing
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class KalshiPrice(BaseModel):
    """Kalshi market price data."""
    price: float  # Yes price (0-1 scale)
    price_cents: int  # Yes price in cents (0-100)
    timestamp: datetime
    market_ticker: str
    market_title: str
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    volume: int
    open_interest: int
    candle_open: float
    candle_direction: str  # "bullish" or "bearish"
    change_from_open: float
    btc_price: float  # BTC reference price from Kalshi (floor_strike)
    btc_price_change: float  # BTC price change percentage
    type: str = "kalshi_price"


class KalshiClient:
    """
    Kalshi API client for fetching market data.
    Uses RSA signing for authentication.
    """

    # API endpoints (updated March 2024)
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    # Demo/paper trading endpoints
    DEMO_BASE_URL = "https://demo-api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        private_key_path: str = None,
        api_key_id: str = None,
        use_demo: bool = False
    ):
        """
        Initialize Kalshi client.

        Args:
            private_key_path: Path to RSA private key PEM file
            api_key_id: Kalshi API key ID
            use_demo: Use demo/paper trading API
        """
        self.private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")
        self.api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID", "")
        self.use_demo = use_demo or os.getenv("KALSHI_USE_DEMO", "").lower() == "true"

        self.base_url = self.DEMO_BASE_URL if self.use_demo else self.BASE_URL

        self._private_key = None
        self._session: Optional[aiohttp.ClientSession] = None

        # Load private key if available
        self._load_private_key()

    def _load_private_key(self):
        """Load RSA private key from file."""
        try:
            if os.path.exists(self.private_key_path):
                with open(self.private_key_path, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )
                logger.info(f"Loaded Kalshi private key from {self.private_key_path}")
            else:
                logger.warning(f"Kalshi private key not found at {self.private_key_path}")
        except Exception as e:
            logger.error(f"Failed to load Kalshi private key: {e}")

    def _sign_request(self, timestamp: str, method: str, path: str) -> str:
        """
        Sign a request using RSA-PSS (matching official Kalshi SDK).

        Args:
            timestamp: Unix timestamp in milliseconds as string
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., /trade-api/v2/markets)

        Returns:
            Base64 encoded signature
        """
        if not self._private_key:
            return ""

        # Message format: timestamp + method + path
        message = f"{timestamp}{method}{path}"

        # Sign with RSA-PSS (required by Kalshi API)
        signature = self._private_key.sign(
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )

        return base64.b64encode(signature).decode('utf-8')

    def _get_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """Generate authentication headers for Kalshi API."""
        if not self._private_key or not self.api_key_id:
            return {"Content-Type": "application/json"}

        timestamp = str(int(time.time() * 1000))
        signature = self._sign_request(timestamp, method, path)

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_markets(
        self,
        series_ticker: str = None,
        status: str = "open",
        limit: int = 100
    ) -> List[Dict]:
        """
        Get list of markets.

        Args:
            series_ticker: Filter by series (e.g., "KXBTC" for Bitcoin)
            status: Market status filter
            limit: Max results
        """
        await self._ensure_session()

        path = "/trade-api/v2/markets"
        params = {"limit": limit, "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker

        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("GET", path)

        try:
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("markets", [])
                else:
                    text = await resp.text()
                    logger.error(f"Failed to get markets: {resp.status} - {text[:200]}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    async def get_market(self, ticker: str) -> Optional[Dict]:
        """Get details for a specific market."""
        await self._ensure_session()

        path = f"/trade-api/v2/markets/{ticker}"
        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("GET", path)

        try:
            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("market")
                else:
                    logger.debug(f"Failed to get market {ticker}: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching market {ticker}: {e}")
            return None

    async def get_events(self, series_ticker: str = None, status: str = "open") -> List[Dict]:
        """Get list of events."""
        await self._ensure_session()

        path = "/trade-api/v2/events"
        params = {"status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker

        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("GET", path)

        try:
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("events", [])
                else:
                    return []
        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return []

    async def search_btc_markets(self) -> List[Dict]:
        """
        Search for active BTC price markets.
        Prioritizes KXBTC15M (15-minute BTC price movement markets).
        """
        all_markets = []

        # First try the 15-minute BTC price series (best for momentum bot)
        btc_series = ["KXBTC15M", "KXBTCD", "KXBTC", "BTC", "BTCD"]

        for series in btc_series:
            markets = await self.get_markets(series_ticker=series, status="open", limit=50)
            if markets:
                logger.info(f"Found {len(markets)} markets in series {series}")
                all_markets.extend(markets)
                if series == "KXBTC15M" and markets:
                    # KXBTC15M is ideal - prioritize it and return early
                    break

        if not all_markets:
            # Fallback: search all markets
            markets = await self.get_markets(status="open", limit=200)
            btc_keywords = ["btc", "bitcoin", "kxbtc"]
            for m in markets:
                ticker = m.get("ticker", "").lower()
                title = m.get("title", "").lower()
                if any(kw in ticker or kw in title for kw in btc_keywords):
                    all_markets.append(m)

        # Sort by volume and prefer active markets with good liquidity
        all_markets.sort(
            key=lambda m: (
                1 if "KXBTC15M" in m.get("ticker", "") else 0,  # Prioritize 15M markets
                m.get("volume_fp", 0) or m.get("volume", 0) or 0
            ),
            reverse=True
        )

        return all_markets

    # =====================
    # Order Execution Methods
    # =====================

    async def get_balance(self) -> Optional[Dict]:
        """
        Get account balance.

        Returns:
            Dict with balance info including 'balance' (cents), 'payout' (pending)
        """
        await self._ensure_session()

        path = "/trade-api/v2/portfolio/balance"
        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("GET", path)

        try:
            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Balance: ${data.get('balance', 0) / 100:.2f}")
                    return data
                else:
                    text = await resp.text()
                    logger.error(f"Failed to get balance: {resp.status} - {text[:200]}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None

    async def place_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str,  # "buy" or "sell"
        count: int,  # Number of contracts
        type: str = "market",  # "market" or "limit"
        yes_price: int = None,  # Price in cents (1-99) for limit orders
        no_price: int = None,  # Alternative: specify no_price
        expiration_ts: int = None,  # Unix timestamp for order expiration
        client_order_id: str = None,  # Optional client-side order ID
        sell_position_floor: int = None,  # Min contracts to keep after sell
        buy_max_cost: int = None  # Max cost in cents for buy orders
    ) -> Optional[Dict]:
        """
        Place an order on Kalshi.

        Args:
            ticker: Market ticker (e.g., "KXBTC15M-25JAN10T1200-B108250")
            side: "yes" or "no" - which side to trade
            action: "buy" or "sell"
            count: Number of contracts
            type: "market" or "limit"
            yes_price: Limit price in cents (1-99) if using yes side
            no_price: Limit price in cents (1-99) if using no side
            expiration_ts: Order expiration timestamp
            client_order_id: Optional tracking ID
            sell_position_floor: Min position to keep after sell
            buy_max_cost: Max total cost for buy order

        Returns:
            Order details including order_id, status
        """
        if not self._private_key or not self.api_key_id:
            logger.error("Cannot place order: API credentials not configured")
            return None

        await self._ensure_session()

        path = "/trade-api/v2/portfolio/orders"
        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("POST", path)

        # Build order payload
        order = {
            "ticker": ticker,
            "action": action.lower(),
            "side": side.lower(),
            "count": count,
            "type": type.lower()
        }

        # Add optional fields
        if yes_price is not None:
            order["yes_price"] = yes_price
        if no_price is not None:
            order["no_price"] = no_price
        if expiration_ts is not None:
            order["expiration_ts"] = expiration_ts
        if client_order_id:
            order["client_order_id"] = client_order_id
        if sell_position_floor is not None:
            order["sell_position_floor"] = sell_position_floor
        if buy_max_cost is not None:
            order["buy_max_cost"] = buy_max_cost

        try:
            import json as json_lib
            logger.info(f"Placing order: {action} {count}x {side} @ {ticker}")
            logger.info(f"Order payload: {json_lib.dumps(order, indent=2)}")

            async with self._session.post(url, headers=headers, json=order) as resp:
                data = await resp.json()

                if resp.status in (200, 201):
                    order_data = data.get("order", data)
                    logger.info(f"Order placed successfully: {order_data.get('order_id')}")
                    return order_data
                else:
                    error_msg = data.get("error", {}).get("message", str(data))
                    error_details = data.get("error", {}).get("details", "")
                    logger.error(f"Failed to place order: {resp.status} - {error_msg}")
                    logger.error(f"Error details: {error_details}")
                    logger.error(f"Full response: {data}")
                    logger.error(f"Request payload was: {json_lib.dumps(order)}")
                    return {"error": error_msg, "details": error_details, "status_code": resp.status}

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"error": str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: The Kalshi order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if not self._private_key or not self.api_key_id:
            logger.error("Cannot cancel order: API credentials not configured")
            return False

        await self._ensure_session()

        path = f"/trade-api/v2/portfolio/orders/{order_id}"
        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("DELETE", path)

        try:
            async with self._session.delete(url, headers=headers) as resp:
                if resp.status in (200, 204):
                    logger.info(f"Order {order_id} cancelled successfully")
                    return True
                else:
                    text = await resp.text()
                    logger.error(f"Failed to cancel order {order_id}: {resp.status} - {text[:200]}")
                    return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get details of a specific order.

        Args:
            order_id: The Kalshi order ID

        Returns:
            Order details dict
        """
        await self._ensure_session()

        path = f"/trade-api/v2/portfolio/orders/{order_id}"
        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("GET", path)

        try:
            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("order", data)
                else:
                    return None
        except Exception as e:
            logger.error(f"Error fetching order {order_id}: {e}")
            return None

    async def get_orders(
        self,
        ticker: str = None,
        status: str = None,  # "resting", "canceled", "executed"
        limit: int = 100
    ) -> List[Dict]:
        """
        Get list of orders.

        Args:
            ticker: Filter by market ticker
            status: Filter by order status
            limit: Max results

        Returns:
            List of order dicts
        """
        await self._ensure_session()

        path = "/trade-api/v2/portfolio/orders"
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status

        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("GET", path)

        try:
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("orders", [])
                else:
                    return []
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []

    async def get_positions(self, ticker: str = None) -> List[Dict]:
        """
        Get current portfolio positions.

        Args:
            ticker: Filter by market ticker

        Returns:
            List of position dicts with fields:
            - ticker: Market ticker
            - position: Net position (positive=yes, negative=no)
            - market_exposure: Current value in cents
            - realized_pnl: Realized P&L in cents
            - total_traded: Total contracts traded
        """
        await self._ensure_session()

        path = "/trade-api/v2/portfolio/positions"
        params = {}
        if ticker:
            params["ticker"] = ticker

        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("GET", path)

        try:
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    positions = data.get("market_positions", [])
                    logger.debug(f"Retrieved {len(positions)} positions")
                    return positions
                else:
                    text = await resp.text()
                    logger.error(f"Failed to get positions: {resp.status} - {text[:200]}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    async def get_fills(
        self,
        ticker: str = None,
        order_id: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get order fills (executed trades).

        Args:
            ticker: Filter by market ticker
            order_id: Filter by order ID
            limit: Max results

        Returns:
            List of fill dicts with fields:
            - trade_id: Unique fill ID
            - order_id: Associated order ID
            - ticker: Market ticker
            - side: "yes" or "no"
            - action: "buy" or "sell"
            - count: Number of contracts
            - yes_price: Price per contract in cents
            - no_price: Inverse price
            - created_time: Fill timestamp
        """
        await self._ensure_session()

        path = "/trade-api/v2/portfolio/fills"
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id

        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("GET", path)

        try:
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    fills = data.get("fills", [])
                    logger.debug(f"Retrieved {len(fills)} fills")
                    return fills
                else:
                    return []
        except Exception as e:
            logger.error(f"Error fetching fills: {e}")
            return []

    async def search_series_by_keyword(
        self,
        keyword: str,
        statuses: list = None,
        limit: int = 200,
    ) -> List[Dict]:
        """
        Search Kalshi markets by keyword and return unique series info.

        Fetches markets across requested statuses, filters by keyword in
        ticker/title, and returns deduplicated series entries with counts.

        Returns list of dicts: {series_ticker, example_title, market_count}
        """
        if statuses is None:
            statuses = ["open", "closed", "settled"]

        kw = keyword.strip().lower()
        series_map: dict = {}

        for status in statuses:
            markets = await self.get_markets(status=status, limit=limit)
            for m in markets:
                ticker = m.get("ticker", "")
                title = m.get("title", "")
                if kw in ticker.lower() or kw in title.lower():
                    series = ticker.split("-")[0] if "-" in ticker else ticker
                    if series not in series_map:
                        series_map[series] = {"series_ticker": series, "example_title": title, "market_count": 0}
                    series_map[series]["market_count"] += 1

        return sorted(series_map.values(), key=lambda x: -x["market_count"])

    async def get_settled_markets(
        self,
        series_ticker: str,
        min_close_ts: int,
        max_close_ts: int,
        limit: int = 200,
    ) -> List[Dict]:
        """
        Fetch all settled markets for a series within a time range.
        Paginates via cursor until all pages are retrieved.

        Returns list of market dicts with keys:
            ticker, close_time, result ("yes"/"no"), floor_strike, title
        """
        await self._ensure_session()

        all_markets: List[Dict] = []
        cursor: str = ""

        while True:
            path = "/trade-api/v2/markets"
            params: Dict = {
                "series_ticker": series_ticker,
                "status": "settled",
                "min_close_ts": min_close_ts,
                "max_close_ts": max_close_ts,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor

            url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
            headers = self._get_auth_headers("GET", path)

            try:
                async with self._session.get(url, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"get_settled_markets status {resp.status}")
                        break
                    data = await resp.json()
                    markets = data.get("markets", [])
                    all_markets.extend(markets)
                    cursor = data.get("cursor", "")
                    if not cursor or len(markets) < limit:
                        break
            except Exception as exc:
                logger.error(f"Error fetching settled markets: {exc}")
                break

        logger.info(f"Fetched {len(all_markets)} settled markets for {series_ticker}")
        return all_markets

    async def get_historical_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> List[Dict]:
        """
        Fetch candlestick data for a settled Kalshi market.
        Tries the historical endpoint first, falls back to the live endpoint
        for recently settled markets not yet in the historical tier.

        Returns list of candlestick dicts, each with:
            end_period_ts, yes_ask {open, high, low, close} (dollar strings),
            yes_bid {open, high, low, close}, price {open, close, ...}, volume_fp
        """
        await self._ensure_session()

        # Derive series ticker from market ticker (e.g. KXBTC15M-26MAR300930-30 -> KXBTC15M)
        series_ticker = ticker.split("-")[0] if "-" in ticker else ticker

        # Try historical endpoint first, fall back to live
        for endpoint_type in ("historical", "live"):
            if endpoint_type == "historical":
                path = f"/trade-api/v2/historical/markets/{ticker}/candlesticks"
            else:
                path = f"/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks"

            params = {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            }

            url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
            headers = self._get_auth_headers("GET", path)

            try:
                async with self._session.get(url, headers=headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        candles = data.get("candlesticks", [])
                        logger.debug(f"Got {len(candles)} candles for {ticker} via {endpoint_type}")
                        return candles
                    elif resp.status == 404 and endpoint_type == "historical":
                        # Not in historical tier yet, try live endpoint
                        continue
                    else:
                        body = await resp.text()
                        logger.warning(f"get_candlesticks {ticker} ({endpoint_type}) status {resp.status}: {body[:200]}")
                        if endpoint_type == "live":
                            return []
            except Exception as exc:
                logger.error(f"Error fetching candlesticks for {ticker} ({endpoint_type}): {exc}")
                if endpoint_type == "live":
                    return []

        return []

    async def batch_cancel_orders(self, order_ids: List[str] = None, ticker: str = None) -> Dict:
        """
        Cancel multiple orders at once.

        Args:
            order_ids: List of order IDs to cancel (optional)
            ticker: Cancel all orders for this ticker (optional)

        Returns:
            Dict with cancelled order IDs and any errors
        """
        if not self._private_key or not self.api_key_id:
            logger.error("Cannot cancel orders: API credentials not configured")
            return {"error": "Not authenticated"}

        await self._ensure_session()

        path = "/trade-api/v2/portfolio/orders"
        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("DELETE", path)

        params = {}
        if order_ids:
            params["order_ids"] = ",".join(order_ids)
        if ticker:
            params["ticker"] = ticker

        try:
            async with self._session.delete(url, headers=headers, params=params) as resp:
                if resp.status in (200, 204):
                    data = await resp.json() if resp.status == 200 else {}
                    logger.info(f"Batch cancel successful: {data}")
                    return data
                else:
                    text = await resp.text()
                    logger.error(f"Batch cancel failed: {resp.status} - {text[:200]}")
                    return {"error": text}
        except Exception as e:
            logger.error(f"Error in batch cancel: {e}")
            return {"error": str(e)}

    async def get_portfolio_settlements(self, limit: int = 100) -> List[Dict]:
        """
        Get settlement history for positions that have resolved.

        Returns:
            List of settlement dicts with P&L info
        """
        await self._ensure_session()

        path = "/trade-api/v2/portfolio/settlements"
        params = {"limit": limit}

        url = f"{self.base_url.replace('/trade-api/v2', '')}{path}"
        headers = self._get_auth_headers("GET", path)

        try:
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("settlements", [])
                else:
                    return []
        except Exception as e:
            logger.error(f"Error fetching settlements: {e}")
            return []


class KalshiLiveFeed:
    """
    Real-time price feed from Kalshi markets.
    Polls the API for live market data.
    """

    def __init__(
        self,
        private_key_path: str = None,
        api_key_id: str = None,
        market_ticker: str = None,
        poll_interval: float = 1.0,
        use_demo: bool = False
    ):
        """
        Initialize Kalshi live feed.
        """
        self.client = KalshiClient(private_key_path, api_key_id, use_demo)
        self.market_ticker = market_ticker or os.getenv("KALSHI_MARKET_TICKER", "")
        self.poll_interval = poll_interval

        self.current_price: float = 0.0
        self.current_price_cents: int = 0
        self.candle_open: float = 0.0
        self.candle_start_time: Optional[datetime] = None
        self.market_data: Optional[Dict] = None

        self.connected_clients: Set = set()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._prev_btc_price: float = 0.0  # Track previous BTC price for change calculation
        self._last_market_check: Optional[datetime] = None  # Track when we last checked for new markets
        self._market_check_interval: int = 60  # Check for new markets every 60 seconds
        self._price_update_callback = None  # Called with (ticker, price_cents) on each price update

    def set_price_update_callback(self, callback):
        """Set callback to be called with (ticker, price_cents) on each price update."""
        self._price_update_callback = callback

    def add_client(self, websocket):
        """Add a client to receive live price updates."""
        self.connected_clients.add(websocket)

    def remove_client(self, websocket):
        """Remove a client from updates."""
        self.connected_clients.discard(websocket)

    def _get_candle_start_time(self) -> datetime:
        """Get the start time of the current 15-minute candle."""
        now = datetime.now(timezone.utc)
        minutes = (now.minute // 15) * 15
        return now.replace(minute=minutes, second=0, microsecond=0)

    def _should_reset_candle(self) -> bool:
        """Check if we should reset the candle open price."""
        current_candle_start = self._get_candle_start_time()
        if self.candle_start_time is None:
            return True
        return current_candle_start > self.candle_start_time

    def _reset_candle(self):
        """Reset candle data for a new 15-minute period."""
        self.candle_open = self.current_price
        self.candle_start_time = self._get_candle_start_time()
        logger.info(f"New Kalshi candle started at {self.candle_start_time}, open: {self.current_price_cents}¢")

    def _update_price(self, price_cents: int):
        """Update current price from market data."""
        self.current_price_cents = price_cents
        self.current_price = price_cents / 100.0

        if self._should_reset_candle():
            self._reset_candle()
        elif self.candle_open == 0:
            self.candle_open = self.current_price

    def get_kalshi_price(self) -> KalshiPrice:
        """Get the current Kalshi market price data."""
        if self.candle_open > 0:
            change_pct = ((self.current_price - self.candle_open) / self.candle_open) * 100
            direction = "bullish" if self.current_price >= self.candle_open else "bearish"
        else:
            change_pct = 0.0
            direction = "neutral"

        market = self.market_data or {}

        # Parse dollar values to cents for display
        def to_cents(val):
            try:
                return int(float(val or 0) * 100)
            except (ValueError, TypeError):
                return 0

        # Get BTC reference price from Kalshi (floor_strike)
        btc_price = float(market.get("floor_strike", 0) or 0)

        # Calculate BTC price change if we have previous data
        btc_change = 0.0
        if hasattr(self, '_prev_btc_price') and self._prev_btc_price > 0 and btc_price > 0:
            btc_change = ((btc_price - self._prev_btc_price) / self._prev_btc_price) * 100
        if btc_price > 0:
            self._prev_btc_price = btc_price

        return KalshiPrice(
            price=self.current_price,
            price_cents=self.current_price_cents,
            timestamp=datetime.now(timezone.utc),
            market_ticker=self.market_ticker or "N/A",
            market_title=market.get("title", "Connecting to Kalshi..."),
            yes_bid=to_cents(market.get("yes_bid_dollars", 0)),
            yes_ask=to_cents(market.get("yes_ask_dollars", 0)),
            no_bid=to_cents(market.get("no_bid_dollars", 0)),
            no_ask=to_cents(market.get("no_ask_dollars", 0)),
            volume=int(float(market.get("volume_fp", 0) or 0)),
            open_interest=int(float(market.get("open_interest_fp", 0) or 0)),
            candle_open=self.candle_open,
            candle_direction=direction,
            change_from_open=round(change_pct, 4),
            btc_price=btc_price,
            btc_price_change=round(btc_change, 4)
        )

    async def broadcast_price(self):
        """Broadcast current price to all connected clients."""
        if not self.connected_clients:
            return

        price_data = self.get_kalshi_price()
        message = price_data.model_dump_json()

        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.add(client)

        for client in disconnected:
            self.connected_clients.discard(client)

    async def _broadcast_loop(self):
        """Broadcast prices at regular intervals."""
        while self._running:
            try:
                await self.broadcast_price()
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(1)

    async def _auto_detect_market(self) -> bool:
        """Auto-detect the best BTC market to track."""
        logger.info("Auto-detecting BTC market on Kalshi...")

        markets = await self.client.search_btc_markets()

        if markets:
            best_market = markets[0]
            self.market_ticker = best_market.get("ticker", "")
            self.market_data = best_market
            logger.info(f"Auto-detected market: {self.market_ticker} - {best_market.get('title', '')}")
            return True
        else:
            logger.warning("No BTC markets found on Kalshi")
            return False

    def _should_check_for_new_market(self) -> bool:
        """Check if we should look for a new market (every 60 seconds or at 15-min boundaries)."""
        now = datetime.now(timezone.utc)

        # Always check if we don't have a market
        if not self.market_ticker:
            return True

        # Check if enough time has passed since last check
        if self._last_market_check is None:
            return True

        seconds_since_check = (now - self._last_market_check).total_seconds()

        # Check more frequently near 15-minute boundaries (within 30 seconds)
        minutes = now.minute
        seconds = now.second
        near_boundary = (minutes % 15 <= 0 and seconds <= 30) or (minutes % 15 == 14 and seconds >= 30)

        if near_boundary and seconds_since_check >= 10:
            return True

        # Regular check interval
        return seconds_since_check >= self._market_check_interval

    async def _check_and_update_market(self) -> bool:
        """Check for new markets and update if a better one is found."""
        try:
            markets = await self.client.search_btc_markets()

            if markets:
                best_market = markets[0]
                new_ticker = best_market.get("ticker", "")

                # If we found a different (newer) market, switch to it
                if new_ticker and new_ticker != self.market_ticker:
                    old_ticker = self.market_ticker
                    self.market_ticker = new_ticker
                    self.market_data = best_market

                    # Reset candle tracking for new market
                    self.candle_open = 0.0
                    self.candle_start_time = None

                    logger.info(f"Switched to new market: {new_ticker} (was: {old_ticker})")
                    return True

            self._last_market_check = datetime.now(timezone.utc)
            return False

        except Exception as e:
            logger.warning(f"Error checking for new markets: {e}")
            return False

    async def _poll_market(self) -> bool:
        """Poll the market for current prices."""
        try:
            # Periodically check for new markets
            if self._should_check_for_new_market():
                await self._check_and_update_market()
                self._last_market_check = datetime.now(timezone.utc)

            market = await self.client.get_market(self.market_ticker)

            if market:
                self.market_data = market

                # Get yes price - Kalshi returns prices in dollars as strings
                yes_bid_str = market.get("yes_bid_dollars") or market.get("yes_bid") or "0"
                yes_ask_str = market.get("yes_ask_dollars") or market.get("yes_ask") or "0"
                last_price_str = market.get("last_price_dollars") or market.get("last_price") or "0"

                # Convert to cents (prices are in dollars like "0.35")
                try:
                    yes_bid = int(float(yes_bid_str) * 100)
                    yes_ask = int(float(yes_ask_str) * 100)
                    last_price = int(float(last_price_str) * 100)
                except (ValueError, TypeError):
                    yes_bid = yes_ask = last_price = 0

                # Use mid price if both available
                if yes_bid and yes_ask:
                    price_cents = (yes_bid + yes_ask) // 2
                elif yes_ask:
                    price_cents = yes_ask
                elif yes_bid:
                    price_cents = yes_bid
                else:
                    price_cents = last_price

                if price_cents > 0:
                    old_price = self.current_price_cents
                    self._update_price(price_cents)

                    if old_price != price_cents:
                        direction = "🟢" if self.current_price >= self.candle_open else "🔴"
                        logger.debug(f"{direction} Kalshi {self.market_ticker}: {price_cents}¢")

                return True
            return False

        except Exception as e:
            logger.error(f"Error polling market: {e}")
            return False

    async def _poll_loop(self):
        """Main polling loop."""
        # Auto-detect market if not specified
        if not self.market_ticker:
            await self._auto_detect_market()

        if not self.market_ticker:
            logger.error("No market ticker configured - waiting for manual configuration")
            while self._running and not self.market_ticker:
                await asyncio.sleep(30)
                await self._auto_detect_market()

        logger.info(f"Starting Kalshi price polling for {self.market_ticker}")

        poll_count = 0
        last_logged_ticker = ""
        while self._running:
            try:
                await self._poll_market()

                # Update position PnL in bot manager whenever price changes
                if self._price_update_callback and self.market_ticker and self.current_price_cents > 0:
                    await self._price_update_callback(self.market_ticker, self.current_price_cents)

                poll_count += 1

                # Log when market changes or periodically
                ticker_changed = self.market_ticker != last_logged_ticker
                if (poll_count % 30 == 1 or ticker_changed) and self.current_price_cents > 0:
                    if ticker_changed:
                        last_logged_ticker = self.market_ticker
                    direction = "🟢" if self.current_price >= self.candle_open else "🔴"
                    logger.info(
                        f"{direction} Kalshi {self.market_ticker}: {self.current_price_cents}¢ "
                        f"(change: {self.get_kalshi_price().change_from_open:+.2f}%)"
                    )

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll loop error: {e}")
                await asyncio.sleep(5)

    async def start(self):
        """Start the Kalshi live feed."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("Kalshi live price feed started")

    async def stop(self):
        """Stop the Kalshi live feed."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        await self.client.close()
        logger.info("Kalshi live price feed stopped")


# Global instance
kalshi_feed = KalshiLiveFeed(
    private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem"),
    api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
    market_ticker=os.getenv("KALSHI_MARKET_TICKER", ""),
    poll_interval=float(os.getenv("KALSHI_POLL_INTERVAL", "1.0")),
    use_demo=os.getenv("KALSHI_USE_DEMO", "").lower() == "true"
)
