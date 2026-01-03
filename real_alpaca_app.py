#!/usr/bin/env python3
"""
ULTIMATE AI TRADING SYSTEM - REAL ALPACA TRADING
Multi-Account Support + Real Balance + Actual Trade Execution
"""

import os, sys, asyncio, threading, time, json, sqlite3, random, re
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import logging
from typing import Dict, List, Any
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global state
trading_active = False
trading_thread = None
trade_history = []
ai_responses = []
hidden_gems = []
accounts = {}
active_account = None

# MULTIPLE ALPACA ACCOUNTS
ALPACA_ACCOUNTS = {
    "account1": {
        "name": "Primary Account",
        "api_key": os.getenv("ALPACA_API_KEY", "PKW3SRTBKRD5ILOE3XVH4QNJVC"),
        "secret_key": os.getenv("ALPACA_SECRET_KEY", "5EcmPxHAJRwfaqrxtNskyp7bqNCWvXkHR6MShBKqqppE"),
        "base_url": "https://paper-api.alpaca.markets"
    },
    "account2": {
        "name": "Secondary Account",
        "api_key": os.getenv("ALPACA_API_KEY_2", ""),
        "secret_key": os.getenv("ALPACA_SECRET_KEY_2", ""),
        "base_url": "https://paper-api.alpaca.markets"
    },
    "account3": {
        "name": "Tertiary Account",
        "api_key": os.getenv("ALPACA_API_KEY_3", ""),
        "secret_key": os.getenv("ALPACA_SECRET_KEY_3", ""),
        "base_url": "https://paper-api.alpaca.markets"
    }
}

# ALL AI MODELS
AI_MODELS = {
    "groq": {
        "models": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b", "qwen-2.5-coder-32b", "mixtral-8x7b-32768"],
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "endpoint": "https://api.groq.com/openai/v1/chat/completions"
    },
    "openai": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "endpoint": "https://api.openai.com/v1/chat/completions"
    },
    "anthropic": {
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "endpoint": "https://api.anthropic.com/v1/messages"
    },
    "google": {
        "models": ["gemini-2.0-flash-exp", "gemini-2.0-flash"],
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models"
    },
    "openrouter": {
        "models": ["x-ai/grok-2-1212", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-exp:free", "deepseek/deepseek-chat"],
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "endpoint": "https://openrouter.ai/api/v1/chat/completions"
    },
    "mistral": {
        "models": ["mistral-large-latest", "mistral-medium-latest"],
        "api_key": os.getenv("MISTRAL_API_KEY", ""),
        "endpoint": "https://api.mistral.ai/v1/chat/completions"
    },
    "deepseek": {
        "models": ["deepseek-chat", "deepseek-coder"],
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "endpoint": "https://api.deepseek.com/chat/completions"
    }
}

THINKING_STYLES = [
    "deductive_reasoning", "inductive_reasoning", "abductive_reasoning", "game_theory",
    "behavioral_economics", "technical_analysis", "fundamental_analysis", "sentiment_analysis",
    "pattern_recognition", "machine_learning", "bayesian_inference", "chaos_theory"
]

def init_database():
    conn = sqlite3.connect('real_alpaca_trading.db')
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY, timestamp TEXT, account TEXT, symbol TEXT, action TEXT,
        quantity REAL, price REAL, order_id TEXT, ai_model TEXT, thinking_style TEXT, confidence REAL
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS ai_calls (
        id INTEGER PRIMARY KEY, timestamp TEXT, model TEXT, thinking_style TEXT,
        response TEXT, confidence REAL, trade_decision TEXT
    )''')
    
    conn.commit()
    conn.close()

async def get_alpaca_account_info(account_id: str):
    """Get real Alpaca account balance and positions"""
    account_config = ALPACA_ACCOUNTS[account_id]
    
    if not account_config['api_key']:
        return None
    
    headers = {
        'APCA-API-KEY-ID': account_config['api_key'],
        'APCA-API-SECRET-KEY': account_config['secret_key']
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get account info
            async with session.get(f"{account_config['base_url']}/v2/account", headers=headers) as resp:
                if resp.status == 200:
                    account_data = await resp.json()
                    
                    # Get positions
                    async with session.get(f"{account_config['base_url']}/v2/positions", headers=headers) as pos_resp:
                        positions = []
                        if pos_resp.status == 200:
                            positions = await pos_resp.json()
                        
                        return {
                            'account_id': account_id,
                            'name': account_config['name'],
                            'balance': float(account_data['cash']),
                            'equity': float(account_data['equity']),
                            'buying_power': float(account_data['buying_power']),
                            'positions': {p['symbol']: float(p['qty']) for p in positions},
                            'portfolio_value': float(account_data['portfolio_value'])
                        }
    except Exception as e:
        logger.error(f"Error fetching account {account_id}: {e}")
    
    return None

async def execute_real_alpaca_trade(account_id: str, symbol: str, action: str, confidence: float, ai_model: str, thinking_style: str):
    """Execute REAL trade on Alpaca"""
    account_config = ALPACA_ACCOUNTS[account_id]
    
    if not account_config['api_key']:
        logger.error(f"No API key for account {account_id}")
        return False
    
    headers = {
        'APCA-API-KEY-ID': account_config['api_key'],
        'APCA-API-SECRET-KEY': account_config['secret_key']
    }
    
    try:
        # Get current price
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{account_config['base_url']}/v2/stocks/{symbol}/trades/latest", headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    current_price = float(data['trade']['p'])
                    
                    # Get account info
                    account_info = await get_alpaca_account_info(account_id)
                    if not account_info:
                        return False
                    
                    # Calculate position size based on confidence
                    max_position_value = account_info['buying_power'] * 0.1 * confidence  # Max 10% per trade, scaled by confidence
                    quantity = max(1, int(max_position_value / current_price))
                    
                    # Place order
                    order_data = {
                        "symbol": symbol,
                        "qty": quantity,
                        "side": "buy" if action == "BUY" else "sell",
                        "type": "market",
                        "time_in_force": "day"
                    }
                    
                    async with session.post(f"{account_config['base_url']}/v2/orders", headers=headers, json=order_data) as order_resp:
                        if order_resp.status in [200, 201]:
                            order_result = await order_resp.json()
                            
                            # Log trade
                            trade_record = {
                                'timestamp': datetime.now().isoformat(),
                                'account': account_id,
                                'symbol': symbol,
                                'action': action,
                                'quantity': quantity,
                                'price': current_price,
                                'order_id': order_result['id'],
                                'ai_model': ai_model,
                                'thinking_style': thinking_style,
                                'confidence': confidence
                            }
                            
                            trade_history.append(trade_record)
                            
                            # Save to database
                            conn = sqlite3.connect('real_alpaca_trading.db')
                            cursor = conn.cursor()
                            cursor.execute('''INSERT INTO trades (timestamp, account, symbol, action, quantity, price, order_id, ai_model, thinking_style, confidence)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                          (trade_record['timestamp'], account_id, symbol, action, quantity, current_price, order_result['id'], ai_model, thinking_style, confidence))
                            conn.commit()
                            conn.close()
                            
                            logger.info(f"✅ REAL TRADE EXECUTED on {account_id}: {action} {quantity} {symbol} at ${current_price:.2f} | Order ID: {order_result['id']}")
                            return True
                        else:
                            error_text = await order_resp.text()
                            logger.error(f"Order failed: {error_text}")
                            
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
    
    return False

async def gather_market_data():
    """Gather market data"""
    data = {'stocks': {}}
    
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'META', 'GOOGL', 'AMZN', 'PLTR']
    
    for symbol in symbols:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        stock_data = await resp.json()
                        result = stock_data['chart']['result'][0]
                        data['stocks'][symbol] = {
                            'price': result['meta']['regularMarketPrice'],
                            'change': result['meta'].get('regularMarketChangePercent', 0)
                        }
        except: pass
    
    return data

async def call_ai_model(provider: str, model: str, thinking_style: str, market_data: Dict, account_info: Dict) -> Dict:
    """Call AI model for trading decision"""
    try:
        config = AI_MODELS[provider]
        if not config['api_key']:
            return {"success": False, "error": "No API key"}
        
        prompt = f"""AI TRADING AGENT using {thinking_style}

REAL ACCOUNT: {account_info['name']}
Balance: ${account_info['balance']:.2f}
Equity: ${account_info['equity']:.2f}
Buying Power: ${account_info['buying_power']:.2f}
Positions: {account_info['positions']}

MARKET DATA:
{json.dumps(market_data['stocks'], indent=1)}

Pick TOP asset to trade NOW. Respond JSON:
{{
  "asset": "SYMBOL",
  "action": "BUY",
  "confidence": 0.85,
  "reasoning": "why this trade"
}}"""
        
        headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
        data = {"messages": [{"role": "user", "content": prompt}], "model": model, "max_tokens": 500, "temperature": 0.7}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config['endpoint'], headers=headers, json=data, timeout=30) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    content = result['choices'][0]['message']['content']
                    
                    try:
                        match = re.search(r'\{.*\}', content, re.DOTALL)
                        if match:
                            ai_decision = json.loads(match.group())
                            return {"success": True, "model": f"{provider}/{model}", "thinking_style": thinking_style, "decision": ai_decision, "raw_response": content}
                    except: pass
                    
                    return {"success": True, "model": f"{provider}/{model}", "thinking_style": thinking_style, "decision": {"asset": "SPY", "action": "BUY", "confidence": 0.5, "reasoning": content[:100]}, "raw_response": content}
                    
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    return {"success": False}

async def ultimate_trading_loop():
    """Real Alpaca trading loop"""
    global trading_active, ai_responses, active_account
    
    logger.info("🚀 REAL ALPACA TRADING SYSTEM ACTIVATED")
    
    # Initialize accounts
    for account_id in ALPACA_ACCOUNTS.keys():
        account_info = await get_alpaca_account_info(account_id)
        if account_info:
            accounts[account_id] = account_info
            if not active_account:
                active_account = account_id
            logger.info(f"✅ {account_info['name']}: ${account_info['portfolio_value']:.2f}")
    
    if not accounts:
        logger.error("❌ No valid Alpaca accounts found")
        return
    
    cycle = 0
    while trading_active:
        cycle += 1
        logger.info(f"\n{'='*20} REAL TRADING CYCLE {cycle} {'='*20}")
        
        try:
            # Refresh account info
            for account_id in list(accounts.keys()):
                account_info = await get_alpaca_account_info(account_id)
                if account_info:
                    accounts[account_id] = account_info
            
            # Get market data
            market_data = await gather_market_data()
            logger.info(f"📊 Market data: {len(market_data['stocks'])} stocks")
            
            # Call AI models
            votes = []
            for _ in range(10):
                provider = random.choice([p for p in AI_MODELS.keys() if AI_MODELS[p]['api_key']])
                model = random.choice(AI_MODELS[provider]['models'])
                thinking_style = random.choice(THINKING_STYLES)
                
                ai_result = await call_ai_model(provider, model, thinking_style, market_data, accounts[active_account])
                
                if ai_result['success']:
                    votes.append(ai_result)
                    ai_responses.append({
                        'timestamp': datetime.now().isoformat(),
                        'model': ai_result['model'],
                        'thinking_style': thinking_style,
                        'decision': ai_result['decision']
                    })
                
                await asyncio.sleep(0.5)
            
            # Democratic voting
            if votes:
                asset_votes = {}
                for vote in votes:
                    asset = vote['decision'].get('asset', 'SPY')
                    if asset not in asset_votes:
                        asset_votes[asset] = []
                    asset_votes[asset].append({
                        'confidence': vote['decision'].get('confidence', 0.5),
                        'model': vote['model'],
                        'thinking_style': vote['thinking_style']
                    })
                
                # Execute top consensus trade
                if asset_votes:
                    best_asset = max(asset_votes.items(), key=lambda x: (len(x[1]), sum(v['confidence'] for v in x[1])))
                    asset, vote_list = best_asset
                    
                    if len(vote_list) >= 3:  # At least 3 AI votes
                        avg_confidence = sum(v['confidence'] for v in vote_list) / len(vote_list)
                        await execute_real_alpaca_trade(active_account, asset, 'BUY', avg_confidence, f"{len(vote_list)}_AIs", "democratic_consensus")
            
            logger.info(f"💰 {accounts[active_account]['name']}: ${accounts[active_account]['portfolio_value']:.2f}")
            logger.info(f"🎯 Cycle {cycle} complete: {len(votes)} AI votes")
            
            await asyncio.sleep(60)  # 1 minute between cycles
            
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            await asyncio.sleep(30)

def run_async_loop():
    asyncio.run(ultimate_trading_loop())

@app.route('/')
def dashboard():
    return render_template('real_alpaca_dashboard.html')

@app.route('/api/accounts')
def get_accounts():
    return jsonify(list(accounts.values()))

@app.route('/api/switch_account', methods=['POST'])
def switch_account():
    global active_account
    data = request.json
    account_id = data.get('account_id')
    if account_id in accounts:
        active_account = account_id
        return jsonify({'status': 'switched', 'account': accounts[account_id]})
    return jsonify({'status': 'error', 'message': 'Account not found'})

@app.route('/api/status')
def get_status():
    current_account = accounts.get(active_account, {'portfolio_value': 0, 'balance': 0, 'positions': {}})
    
    return jsonify({
        'trading_active': trading_active,
        'active_account': active_account,
        'portfolio_value': current_account.get('portfolio_value', 0),
        'balance': current_account.get('balance', 0),
        'positions': current_account.get('positions', {}),
        'total_trades': len(trade_history),
        'ai_calls': len(ai_responses),
        'total_accounts': len(accounts),
        'active_models': sum(1 for m in AI_MODELS.values() if m['api_key'])
    })

@app.route('/api/trades')
def get_trades():
    return jsonify(trade_history[-100:])

@app.route('/api/ai_responses')
def get_ai_responses():
    return jsonify(ai_responses[-200:])

@app.route('/api/start_trading', methods=['POST'])
def start_trading():
    global trading_active, trading_thread
    
    if not trading_active:
        trading_active = True
        trading_thread = threading.Thread(target=run_async_loop)
        trading_thread.daemon = True
        trading_thread.start()
        logger.info("🚀 REAL ALPACA TRADING STARTED")
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/api/stop_trading', methods=['POST'])
def stop_trading():
    global trading_active
    trading_active = False
    logger.info("🛑 REAL ALPACA TRADING STOPPED")
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    init_database()
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("="*60)
    logger.info("💰 REAL ALPACA TRADING SYSTEM - MULTI-ACCOUNT 💰")
    logger.info("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=False)
