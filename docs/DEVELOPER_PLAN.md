# Developer Plan

This document provides a roadmap for our Binance integration and outlines our evolution strategy. Early experiments (V1 and V2) served as initial proofs-of-concept, while V3 is our current production system that we will continuously evolve. The plan also details our directions to expand into V4 (AI-Driven Logic) and beyond.

---

## 1. Overview of Experimental and Production Versions

### Experimental Phases (V1 & V2)

#### V1: Basic Data Retrieval (Experiment)
- **Goal:** Establish a reliable connection to Binance.
- **Features:**
  - Create a file (src/v1_main.py) connecting to Binance using python-binance.
  - Retrieve account info (balances) and open orders for a specified symbol (e.g., TRUMPUSDC).
  - Log the results.
- **Purpose:** Serve as foundational experimentation to understand the Binance API and data retrieval process.

#### V2: Automated SELL at +1% (Experiment)
- **Goal:** Develop automated trading logic and logging.
- **Features:**
  - In src/v2_main.py, poll for recent BUY trades and automatically place a SELL at +1% above the fill price.
  - Include a --dry-run option to simulate SELL orders without executing them.
  - Track old BUY trades on startup to avoid trading loops.
  - Log a 7-period moving average (MA7) and indicate its direction (UP, DOWN, or FLAT).
- **Purpose:** Experiment with automatic trade execution while ensuring that only new trades trigger system actions.

### Production Baseline

#### V3: Full Integration (Production)
- **Goal:** Merge and refine successful experiments into a single, robust system.
- **Features:**
  - Integrates V2 features:
    - Automated SELL orders following new BUY fills (and closing the loop with SELL→BUY logic).
    - Maintains comprehensive order state tracking, distinguishing between manual and system orders.
    - Ring buffer for logging technical indicators like MA7.
    - Supports --dry-run mode for testing.
  - Detects and differentiates new SELL fills, ensuring that only appropriate triggers generate counter trades.
  - Tracks both old and newly placed orders to prevent repeated orders for the same fill.
- **Purpose:** Establish a stable and reliable system for continuous live trading execution on Binance, serving as the baseline for future enhancements.

---

## 2. Basic State Management to Prevent Loops

To avoid infinite trading loops:
- Use an in-memory dictionary or a basic database table with an `origin` field indicating `MANUAL` or `SYSTEM`.
- For each fill event:
  - **BUY:** Trigger a SELL at +1% if the fill is manually placed.
  - **SELL:** Do not trigger a new BUY if the SELL is a system-generated order.
- **Pseudocode Example:**
  ```python
  orders_state = {}  # e.g., order_id -> {'symbol': ..., 'price': ..., 'origin': 'MANUAL'/'SYSTEM'}

  def on_buy_placed_manually(order_id: str, symbol: str, buy_price: float):
      orders_state[order_id] = {
          'symbol': symbol,
          'price': buy_price,
          'origin': 'MANUAL'
      }
      # Place automatic SELL if needed
      sell_price = buy_price * 1.01
      sell_order_id = place_sell_order(symbol, sell_price)
      orders_state[sell_order_id] = {
          'symbol': symbol,
          'price': sell_price,
          'origin': 'SYSTEM'
      }

  def on_order_filled(order_id: str, side: str):
      order_info = orders_state.get(order_id)
      if not order_info:
          return  # Not tracked or not relevant

      if side == 'SELL':
          if order_info['origin'] == 'SYSTEM':
              print("System SELL filled. Doing nothing further.")
              return
          # Optionally place a new BUY if business logic requires it.
  ```

---

## 3. Looking Forward: Expanding into V4 and Beyond

### V4: AI-Driven Logic
As we evolve V3, the next major step is to integrate AI-based strategies:
- **Replace Fixed Logic:** Move away from fixed ±1% thresholds and develop an AI-based strategy module that dynamically determines optimal trade parameters.
- **Order Tracking:** Continue to track each order with `origin = "SYSTEM"` to prevent infinite loops.
- **Adaptive Algorithms:** Research and integrate machine learning techniques capable of recognizing market patterns and predicting favorable entry/exit points.

#### Implementation Plan for V4:
1. **Research & Design:** 
   - Study current AI and ML frameworks that can be integrated for real-time decision-making.
   - Design a modular AI module that can be swapped or updated without altering core trade logic.

2. **Development:**
   - Create a new module (e.g., `src/ai_strategy.py`) dedicated to AI-driven logic.
   - Develop functions that ingest market data, perform pattern recognition, and output trade signals.
   - Integrate this module into the existing trade execution flow of V3.

3. **Testing & Validation:**
   - **Backtesting:** Utilize historical data to validate AI predictions and optimize parameters.
   - **Paper Trading:** Deploy the AI strategy in a simulated environment to ensure reliability.
   - **Performance Metrics:** Continuously monitor win rates, profit/loss metrics, and system responsiveness.

4. **Deployment:**
   - Gradually deploy AI-driven orders alongside the current logic (allowing for fallback if the AI module fails).
   - Log and analyze all trades made by AI-driven logic separately for performance review.
  
### Best Practices for Expanding V3:
- **Modular Code Design:** Keep future enhancements decoupled from core functionality.
- **Incremental Testing:** Validate each new feature in isolation before integrating.
- **Documentation and Logging:** Maintain clear documentation and granular logs to simplify debugging and future development.
- **User Feedback:** Gather and review feedback from early users of V3 to guide enhancements.

---

## 4. Summary
- **Experimental Phases (V1 & V2):** Provided rapid prototyping to explore core API interactions and basic trading logic.
- **V3 (Production Baseline):** A fully integrated system ensuring loop prevention, order tracking, and robust real-time trading logic.
- **Future (V4 and Beyond):** Transition to AI-driven strategies for dynamic and optimized trading decisions, supported by extensive testing and modular design principles.

This plan sets the stage for our continued evolution toward a highly adaptive and profitable trading system.

## Handling Existing Portfolio (Version 2/3 Enhancement)

To avoid placing SELL orders for old BUY fills, we will:

1. On script startup, retrieve all existing BUY trades for the symbol from Binance.  
2. Store their trade IDs (or timestamps) in a set, e.g. old_buy_ids.  
3. During the main loop, ignore any BUY trades whose ID (or timestamp) is in old_buy_ids.  

This way, only new BUY trades (from the moment we start the script) trigger an automatic SELL at +1%. 

## V2 Dry-Run Option

Additionally, our v2_main.py script supports a --dry-run argument:

• If --dry-run is provided, the script will detect BUY fills and compute SELL prices at +1% but will not actually place any orders on Binance.  
• This is useful for testing the flow without risking real trades.  
• Normal usage (without --dry-run) logically places the SELL orders via the Binance API.

## V2: Logging a 7-Period Moving Average (MA7)

We will:
1. Retrieve the last 7 candlesticks (klines) for TRUMPUSDC (e.g., 1m interval).  
2. Calculate the average of the close prices to get MA7.  
3. Compare the new MA7 against the previous iteration's MA7 to determine if it is going up, going down, or unchanged.  
4. Log this in the console (or logs).

Implementation Details:
- Use client.get_klines(symbol, interval="1m", limit=7).  
- Sum the close values and divide by 7.  
- Keep track of the previous MA7 in a variable (previous_ma7).  
- If current_ma7 > previous_ma7 → "MA7 is going up!"  
- If current_ma7 < previous_ma7 → "MA7 is going down!"  
- Otherwise → "No change in MA7."

This is purely for monitoring; it does not affect the trading logic in v2_main.py yet.

```bash
# Example usage:
python src/v2_main.py --dry-run
``` 