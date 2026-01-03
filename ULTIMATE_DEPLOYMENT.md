# ULTIMATE FIGBOT DEPLOYMENT

## What's Included

**Backend**: `democratic_trader_hybrid.py` - Ultimate democratic AI system
- API keys via environment variables (OpenAI, Groq, Mistral, OpenRouter)
- 14 thinking styles per model
- Maximum concurrent API calls
- Comprehensive data logging (predictions, trades, API calls, leaderboard)
- $30 → $10,000 goal with aggressive position sizing
- 1 trade per minute on Alpaca paper trading

**Frontend**: `dashboard.html` - Functional UI (no sliders)
- Art-reactive visuals driven by live trading data
- Real-time stats, trades, predictions, AI calls
- Auto-updating charts and tables
- Start/Stop/Config controls

**API**: `dashboard_api.py` - Data endpoints for UI
**Server**: `hybrid_app.py` - Flask app with threading

## Deploy to Railway

1. **Push to GitHub**:
```bash
cd d:\trading_bot\clean_deploy
git init
git add .
git commit -m "Ultimate FIGBOT with functional UI"
git remote add origin https://github.com/Figbot1/trading-bot.git
git push -u origin main
```

2. **Railway Setup**:
- Connect GitHub repo
- Set environment variables:
  - `APCA_API_KEY_ID=your_key`
  - `APCA_API_SECRET_KEY=your_secret`
  - `AUTO_START=1`

3. **Deploy**: Railway auto-deploys from Procfile

## System Features

✅ **Maximum API Usage**: All 7 APIs × all models × 14 thinking styles = 196 concurrent calls per cycle
✅ **Democratic Voting**: Weighted consensus based on historical accuracy
✅ **Comprehensive Logging**: Every prediction, API call, trade recorded in SQLite
✅ **Leaderboard**: Real-time ranking of AI models by accuracy
✅ **ML Learning**: Agents see their performance history before making predictions
✅ **Multi-Timeframe**: 5s, 30s, 1m, 5m, 15m, 30m, 1h, 3h, 12h, 1d, 3d, 1w, 2w, 1M, 3M, 6M, 1Y
✅ **Aggressive Trading**: Up to 30% position sizing for $30→$10k goal
✅ **Rate Limited**: 1 trade per minute to avoid overtrading
✅ **Art-Reactive UI**: Visuals respond to volatility and activity

## Working API Keys

- OpenAI: 1 key
- Groq: 3 keys
- Mistral: 1 key
- OpenRouter: 2 keys

Total: 7 providers, 14+ models, 196 AI combinations per cycle

## Local Testing

```bash
pip install -r requirements.txt
python hybrid_app.py
# Visit http://localhost:5000
```

## Architecture

```
User → dashboard.html (functional UI)
  ↓
hybrid_app.py (Flask + threading)
  ↓
democratic_trader_hybrid.py (Ultimate AI democracy)
  ↓
- gather_max_data() → stocks, crypto, social media
- call_all_ais_max() → 196 concurrent AI calls
- calculate_consensus() → weighted democratic voting
- execute_trade() → Alpaca paper trading
  ↓
ultimate_democracy.db (SQLite)
  ↓
dashboard_api.py → stats, trades, predictions, leaderboard
```

## Goal

**$30 → $10,000 in 1 week** through maximum AI democracy

Everyone gives what they can, takes what they need.
