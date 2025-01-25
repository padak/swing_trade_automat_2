import os
import time
import math
import argparse
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
from scipy import stats
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import SGDRegressor

def main():
    parser = argparse.ArgumentParser(description="Trend Detector v1 - single transaction, with swing threshold.")
    parser.add_argument("--symbol", default="TRUMPUSDC", help="Symbol to trade (default: TRUMPUSDC)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Significance threshold for slope magnitude')
    args = parser.parse_args()

    load_dotenv()  # Load environment if needed for e.g. BINANCE_API_KEY, BINANCE_API_SECRET

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("Warning: BINANCE_API_KEY / BINANCE_API_SECRET not found in .env - continuing read-only market data.")
    client = Client(api_key, api_secret)

    symbol = args.symbol

    # ------------------------------------------------------------------------
    # 1) On startup, fetch the *latest* close price so we can set initial TRUMP
    #    We'll pretend we have 500 USDC and 500 USDC worth of TRUMP.
    # ------------------------------------------------------------------------
    start_klines = client.get_klines(symbol=symbol, interval="1m", limit=1)
    if len(start_klines) < 1:
        print("Could not fetch initial price, script will exit.")
        return

    # last close price
    startup_price = float(start_klines[-1][4])
    print(f"Fetched startup price={startup_price:.4f} for {symbol} to initialize balances.")

    # We'll have 500 USDC and the equivalent 500 USDC worth of TRUMP
    usdc_balance = 500.0
    trump_balance = 500.0 / startup_price  # get some TRUMP so we can also test sells
    total_profit_usdc = 0.0  # realized net profit from trades (above the initial holding's cost basis)

    print(f"Initial USDC={usdc_balance:.2f}, TRUMP={trump_balance:.6f} (worth ~500 USDC), total virtual capital=~1000 USDC")

    # Poll interval
    poll_interval_sec = 1

    # Introduce a swing threshold. E.g. 1% difference from our last entry price
    swing_threshold = 0.01  # 1% swing required before next trade
    last_entry_price = startup_price  # track price where we last bought/sold
    last_action = None  # 'BUY' or 'SELL'; helps avoid multiple trades in same direction

    print("Strategy: Single Transaction per Down/Up with a 1% swing threshold.")
    print("Press Ctrl+C to stop.\n")

    # Add to main initialization
    trading_model = TradingModel()

    try:
        while True:
            try:
                klines = client.get_klines(symbol=symbol, interval="1m", limit=25)
                if len(klines) < 25:
                    print("Not enough klines to compute MAs, waiting...")
                    time.sleep(poll_interval_sec)
                    continue

                close_prices = [float(k[4]) for k in klines]
                short_ma = sum(close_prices[-7:]) / 7.0  # MA7
                long_ma  = sum(close_prices) / 25.0      # MA25
                current_price = close_prices[-1]

                # Determine trend
                if short_ma > long_ma:
                    trend = "UP"
                elif short_ma < long_ma:
                    trend = "DOWN"
                else:
                    trend = "FLAT"

                # Time label for logs
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{now_str}] MA7={short_ma:.5f} / MA25={long_ma:.5f} => Trend={trend}, Price={current_price:.5f}"
                )

                # Check if price moved enough from our last entry
                price_change_ratio = abs(current_price - last_entry_price) / last_entry_price
                if price_change_ratio < swing_threshold:
                    # Not enough swing to justify a new trade
                    total_trump_value = trump_balance * current_price
                    total_equity = usdc_balance + total_trump_value
                    print(
                        f"[{now_str}] Swing < {swing_threshold*100:.1f}% threshold, no trade. "
                        f"usdc={usdc_balance:.2f}, trump={trump_balance:.6f} (~{total_trump_value:.2f} USDC), "
                        f"realizedProfit={total_profit_usdc:.2f}, totalEquity={total_equity:.2f}\n"
                    )
                    time.sleep(poll_interval_sec)
                    continue

                # If trend = UP => single BUY
                if trend == "UP" and last_action != "BUY":
                    if usdc_balance > 1.2:
                        used_usdc = min(usdc_balance, 200.0)  # example: choose up to 200 USDC
                        if used_usdc < 1.2:
                            print(f"[{now_str}] Skipping BUY - below 1.2 USDC threshold.")
                        else:
                            fee = used_usdc * 0.001
                            spend_usdc = used_usdc - fee
                            if spend_usdc > 0:
                                bought_amount = spend_usdc / current_price
                                trump_balance += bought_amount
                                usdc_balance -= used_usdc
                                last_entry_price = current_price
                                last_action = "BUY"
                                print(f"[{now_str}] Sim-BUY {bought_amount:.6f} TRUMP at {current_price:.5f}, spending {used_usdc:.2f} inc fee={fee:.2f}")

                # If trend = DOWN => single SELL
                elif trend == "DOWN" and last_action != "SELL":
                    if trump_balance > 0.000001:
                        gross_usdc = trump_balance * current_price
                        if gross_usdc < 1.2:
                            print(f"[{now_str}] Skipping SELL - below 1.2 USDC threshold.")
                        else:
                            fee = gross_usdc * 0.001
                            net_usdc = gross_usdc - fee

                            # For realized profit, we need cost basis. We'll skip it or do naive approach:
                            # realizedProfit remains 0 unless we track average cost.
                            usdc_balance += net_usdc
                            trump_balance = 0.0
                            last_entry_price = current_price
                            last_action = "SELL"
                            print(f"[{now_str}] Sim-SELL ALL TRUMP at {current_price:.5f}, gross={gross_usdc:.2f}, fee={fee:.2f}, net={net_usdc:.2f} USDC")

                total_trump_value = trump_balance * current_price
                total_equity = usdc_balance + total_trump_value
                print(
                    f"[{now_str}] usdc={usdc_balance:.2f}, trump={trump_balance:.6f} (~{total_trump_value:.2f} USDC), "
                    f"realizedProfit={total_profit_usdc:.2f}, totalEquity={total_equity:.2f}\n"
                )

                time.sleep(poll_interval_sec)

            except (BinanceAPIException, BinanceRequestException) as e:
                print(f"Error fetching market data or parsing results: {e}")
                time.sleep(5)

    except KeyboardInterrupt:
        final_trump_value = trump_balance * current_price
        final_total = usdc_balance + final_trump_value
        print("\nShutting down read-only simulation...")
        print(f"Final USDC={usdc_balance:.2f}, TRUMP={trump_balance:.6f} (~{final_trump_value:.2f} USDC), total equity={final_total:.2f} USDC")
        print(f"Realized net profit recorded={total_profit_usdc:.2f} USDC")

def detect_trends(input_file):
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  # Add explicit format
        # ... existing code ...
        result = {
            'trend_direction': trend_direction,
            'slope': slope,
            'p_value': p_value,
            'r_squared': model.score(X, y),  # New metric
            'confidence_interval': {
                'lower': conf_int[0][0],
                'upper': conf_int[0][1]
            }
        }
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return None
    except pd.errors.EmptyDataError:
        print("Error: Empty CSV file")
        return None

def check_normality(residuals):
    """Check normality using Shapiro-Wilk test"""
    stat, p = stats.shapiro(residuals)
    return p > 0.05

class SmartTrendDetector:
    def __init__(self, data, risk_per_trade=0.02, atr_multiplier=3, rsi_window=14, ema_short=20, ema_long=50):
        self.data = data
        self.risk_per_trade = risk_per_trade  # Risk 2% of capital per trade
        self.atr_multiplier = atr_multiplier
        self.rsi_window = rsi_window
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.account_balance = 10000  # Starting balance
        self.position_size = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # Calculate indicators
        self._calculate_indicators()
        
    def _calculate_indicators(self):
        # Core trend indicators
        self.data['EMA_short'] = ta.ema(self.data['close'], length=self.ema_short)
        self.data['EMA_long'] = ta.ema(self.data['close'], length=self.ema_long)
        
        # Momentum and volatility indicators
        self.data['RSI'] = ta.rsi(self.data['close'], length=self.rsi_window)
        self.data['ATR'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=14)
        
        # Volume analysis
        self.data['VWAP'] = ta.vwap(self.data['high'], self.data['low'], self.data['close'], self.data['volume'])
        self.data['OBV'] = ta.obv(self.data['close'], self.data['volume'])
        
        # Advanced indicators
        self.data['ADX'] = ta.adx(self.data['high'], self.data['low'], self.data['close'], length=14)['ADX_14']
        self.data['MACD'] = ta.macd(self.data['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
        self.data['BBANDS_upper'], self.data['BBANDS_mid'], self.data['BBANDS_lower'] = ta.bbands(
            self.data['close'], length=20, std=2)
        
    def analyze(self):
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        # Trend direction logic
        ema_bullish = latest['EMA_short'] > latest['EMA_long'] and prev['EMA_short'] <= prev['EMA_long']
        ema_bearish = latest['EMA_short'] < latest['EMA_long'] and prev['EMA_short'] >= prev['EMA_long']
        
        # Trend strength filter
        strong_trend = latest['ADX'] > 25
        weak_trend = latest['ADX'] < 20
        
        # Volume confirmation
        volume_spike = latest['volume'] > 1.5 * self.data['volume'].rolling(20).mean().iloc[-1]
        obv_confirmation = latest['OBV'] > self.data['OBV'].rolling(20).mean().iloc[-1]
        
        # Risk management calculations
        atr = latest['ATR']
        price = latest['close']
        self.position_size = (self.account_balance * self.risk_per_trade) / (self.atr_multiplier * atr)
        self.stop_loss = price - self.atr_multiplier * atr if ema_bullish else price + self.atr_multiplier * atr
        self.take_profit = price + 2 * self.atr_multiplier * atr if ema_bullish else price - 2 * self.atr_multiplier * atr
        
        # Generate signals
        signal = "HOLD"
        confidence = 0
        
        # Trend following entry
        if ema_bullish and strong_trend and volume_spike and obv_confirmation:
            if latest['RSI'] < 70 and latest['close'] > latest['VWAP']:
                signal = "BUY"
                confidence = min(90, latest['ADX'] / 0.4)
        elif ema_bearish and strong_trend and volume_spike and obv_confirmation:
            if latest['RSI'] > 30 and latest['close'] < latest['VWAP']:
                signal = "SELL"
                confidence = min(90, latest['ADX'] / 0.4)
                
        # Mean reversion strategy during weak trends
        if weak_trend:
            if latest['close'] < latest['BBANDS_lower'] and latest['RSI'] < 35:
                signal = "BUY"
                confidence = 65
            elif latest['close'] > latest['BBANDS_upper'] and latest['RSI'] > 65:
                signal = "SELL"
                confidence = 65
                
        # Add divergence detection
        price_higher_high = latest['close'] > prev['close'] and latest['high'] > prev['high']
        rsi_lower_high = latest['RSI'] < prev['RSI']
        if price_higher_high and rsi_lower_high and latest['RSI'] > 70:
            signal = "SELL"
            confidence = 75
            
        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "position_size": round(self.position_size, 2),
            "risk_reward_ratio": 2.0
        }

# Example usage:
# data = pd.read_csv('price_data.csv')
# detector = SmartTrendDetector(data)
# print(detector.analyze())

class TradingModel:
    def __init__(self):
        self.model = SGDRegressor()
        self.last_state = None
        self.last_action = None
        
    def predict_threshold(self, market_state):
        """Predict optimal threshold using learned model"""
        if not hasattr(self.model, "coef_"):
            # Initial random exploration
            return max(0.1, min(np.random.normal(1.0, 0.5), 3.0))
        return max(0.1, self.model.predict([market_state])[0])

    def update_model(self, reward, new_state):
        """Reinforcement learning update"""
        if self.last_state is not None:
            self.model.partial_fit([self.last_state], [reward])
        self.last_state = new_state

def get_market_state(client, symbol):
    """Create normalized market state vector"""
    klines = client.get_klines(symbol=symbol, interval="1h", limit=24)
    prices = np.array([float(k[4]) for k in klines])
    return np.array([
        prices.mean(),          # 24h average
        prices.std(),           # Volatility
        prices[-1]/prices[0],   # 24h return
        (prices[-1] - min(prices))/(max(prices) - min(prices))  # Relative position
    ])

def user_friendly_alert(action, reason):
    """Generate plain language alerts"""
    messages = {
        "BUY": f"📈 Buying opportunity detected! {reason}",
        "SELL": f"📉 Selling recommended! {reason}",
        "HOLD": "🛑 No clear trend - maintaining position"
    }
    print(f"\n⚠️ ALERT: {messages[action]}\n")

def execute_trade(client, action, amount):
    """Execute trade with plain language confirmation"""
    if action == "BUY":
        print(f"🟢 Purchasing {amount:.2f} USDC worth of assets")
        # client.order_market_buy(...)
    elif action == "SELL":
        print(f"🔴 Liquidating {amount:.2f} USDC equivalent position")
        # client.order_market_sell(...)
    else:
        print("🟡 No action taken - maintaining current position")

if __name__ == "__main__":
    main()
