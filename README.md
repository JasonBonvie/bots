# BTC/Kalshi 15-Minute Momentum Bot Dashboard

A full-stack real-time trading signals dashboard for BTC momentum analysis with Kalshi binary options integration and automated trading capabilities.

## Features

- **Real-time BTC Price Tracking**: Live price data from TradingView (Binance BTCUSDT)
- **Technical Indicators**: RSI(14), EMA(8/21), MACD(12,26,9), Close position
- **Momentum Scoring**: Automated scoring system (0-4) for UP/DOWN signals
- **Kalshi Integration**: Live contract prices, order execution, position tracking
- **Auto-Trading**: Automated position opening based on signal strength
- **Bot Management**: Create, start, pause, stop bots with configurable strategies
- **WebSocket Updates**: Auto-updates every 15 minutes synced to candle close
- **Edge Math Calculator**: Interactive martingale sequence and EV analysis
- **Dark Terminal UI**: Professional trading terminal aesthetic

## Stack

- **Backend**: Python (FastAPI + WebSockets)
- **Data**: TradingView (BINANCE:BTCUSDT OHLCV)
- **Indicators**: pandas-ta
- **Trading**: Kalshi API with RSA authentication
- **Frontend**: Single HTML file with IBM Plex fonts

## Project Structure

```
nekalshi/
├── main.py              # FastAPI application
├── bot/
│   ├── __init__.py
│   ├── signals.py       # Indicator computation
│   ├── scheduler.py     # 15-min candle sync + auto-trade
│   ├── models.py        # Pydantic models
│   ├── bot_manager.py   # Bot CRUD + position management
│   ├── kalshi_feed.py   # Kalshi API client + order execution
│   └── live_price.py    # Live BTC price feed
├── static/
│   └── index.html       # Dashboard frontend
├── requirements.txt
├── .env.example         # Environment variable template
└── README.md
```

## Setup

1. **Create virtual environment** (recommended):
   ```bash
   cd nekalshi
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kalshi API** (optional, for live trading):
   ```bash
   cp .env.example .env
   # Edit .env with your Kalshi credentials
   ```

4. **Run the server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. **Open the dashboard**:
   Navigate to http://localhost:8000 in your browser

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `KALSHI_API_KEY_ID` | Kalshi API key ID | For trading |
| `KALSHI_PRIVATE_KEY_PATH` | Path to RSA private key PEM file | For trading |
| `KALSHI_USE_DEMO` | Use demo/paper trading API (`true`/`false`) | No |
| `KALSHI_LIVE_TRADING` | Enable real order execution (`true`/`false`) | No |
| `KALSHI_MARKET_TICKER` | Override auto-detected market ticker | No |

## API Endpoints

### Signals & Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the dashboard HTML |
| `/api/signals` | GET | Returns latest computed signals |
| `/api/refresh` | GET | Force refresh signals |
| `/api/health` | GET | Health check with next candle time |
| `/api/live_price` | GET | Current BTC price with candle info |
| `/api/chart_data` | GET | 100-candle OHLCV history for chart |
| `/ws` | WebSocket | Real-time signal & price broadcasts |

### Kalshi Market Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/kalshi_price` | GET | Current Kalshi contract price |
| `/api/kalshi_markets` | GET | Available BTC markets |
| `/api/kalshi/balance` | GET | Account balance |
| `/api/kalshi/positions` | GET | Open positions from Kalshi |
| `/api/kalshi/orders` | GET | Order history |
| `/api/kalshi/fills` | GET | Trade fill history |

### Kalshi Trading

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/kalshi/order` | POST | Place an order on Kalshi |
| `/api/kalshi/order/{id}` | DELETE | Cancel an order |
| `/api/kalshi/sync` | POST | Sync local positions with Kalshi |

### Bot Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/bots` | GET | List all bots with summary |
| `/api/bots` | POST | Create a new bot |
| `/api/bots/{id}` | GET | Get bot details with metrics |
| `/api/bots/{id}` | DELETE | Delete a bot |
| `/api/bots/{id}/start` | POST | Start a bot |
| `/api/bots/{id}/pause` | POST | Pause a bot |
| `/api/bots/{id}/stop` | POST | Stop a bot |
| `/api/bots/{id}/auto_trade` | POST | Manually trigger auto-trade |

### Position Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/positions` | GET | List open positions |
| `/api/positions` | POST | Open a new position |
| `/api/positions/{id}/close` | POST | Close a position |
| `/api/trades` | GET | Trade history with P&L |

### Trading Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trading/status` | GET | Current trading system status |
| `/api/trading/auto_trade` | POST | Enable/disable auto-trading |

## Signal Scoring System

Each indicator contributes to UP and DOWN scores (4 filters total):

| Indicator | UP Condition | DOWN Condition | Points |
|-----------|--------------|----------------|--------|
| EMA 8/21 | EMA8 > EMA21 | EMA8 < EMA21 | +1.0 |
| RSI (14) | 50 < RSI < 70 | 30 < RSI < 50 | +1.0 |
| MACD | MACD > Signal | MACD < Signal | +1.0 |
| Close Pos | > 0.65 | < 0.35 | +1.0 |

**Direction**: UP if score_up ≥ 3, DOWN if score_dn ≥ 3, else SKIP

## Auto-Trading

When auto-trading is enabled and a bot is running:

1. Every 15 minutes at candle close, signals are computed
2. If signal is UP or DOWN (not SKIP), eligible bots open positions
3. Position side: YES for UP signals, NO for DOWN signals
4. Stake calculated based on bot config (base stake × martingale multiplier)

### Risk Management

Bots support configurable risk limits:

- **Daily Loss Limit**: Max loss before bot stops trading for the day
- **Daily Trade Limit**: Max number of trades per day
- **Max Position Size**: Maximum contracts per trade
- **Martingale**: Optional progressive stake sizing after losses

## Dashboard Tabs

### Tab 1: Strategy
- Live BTC price with 15-minute candle direction
- Live Kalshi contract price with bid/ask spread
- Visual signal bars with bullish/bearish tags
- 4-step decision logic flow
- Entry badge with confidence level

### Tab 2: Bots
- Bot creation and management
- Start/pause/stop controls
- Position tracking with unrealized P&L
- Trade history with metrics

### Tab 3: Edge Math
- Interactive sliders for win rate, boost, bankroll, stake, stops
- Expected Value (EV) visualization bars
- Martingale sequence table
- Improvement statistics

### Tab 4: Bot Code
- Syntax-highlighted signal computation code
- Implementation reference

## WebSocket Protocol

Connect to `ws://localhost:8000/ws` to receive:

- **On connect**: Latest cached signal + live price + Kalshi price
- **Every 1.5s**: Live BTC price update (`type: "live_price"`)
- **Every 0.5s**: Kalshi contract price update (`type: "kalshi_price"`)
- **Every 15 minutes**: New signal JSON at :00, :15, :30, :45 + 2s buffer
- **Send "refresh"**: Request immediate signal update

## Kalshi Order Execution

### Paper Trading (Default)

By default, orders are simulated locally. Positions are tracked in-memory with realistic P&L calculations.

### Live Trading

To enable real order execution:

1. Generate an RSA key pair for Kalshi API
2. Add your API credentials to `.env`:
   ```
   KALSHI_API_KEY_ID=your-api-key-id
   KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
   KALSHI_LIVE_TRADING=true
   ```
3. Restart the server

⚠️ **Warning**: Live trading executes real orders with real money. Test thoroughly in paper mode first.

### Order Types

- **Market Orders**: Execute immediately at best available price
- **Limit Orders**: Execute only at specified price or better

## Configuration

Default settings (modify in source or via environment):

- Exchange: TradingView BINANCE:BTCUSDT (primary)
- Timeframe: 15 minutes
- OHLCV bars: 50
- Rate limit retries: 3
- Candle close buffer: 2 seconds
- Live price poll interval: 1.5 seconds
- Kalshi price poll interval: 0.5 seconds
- Market auto-detection interval: 60 seconds

## Development

### Running Tests

```bash
pytest tests/
```

### Database Migration (TODO)

Currently using in-memory storage. For production:

1. Add SQLAlchemy models
2. Run migrations with Alembic
3. Update `bot_manager.py` to use database

## License

MIT