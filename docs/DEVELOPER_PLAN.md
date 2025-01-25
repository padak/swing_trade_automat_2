# Developer Plan

This document provides a simplified roadmap for our Binance integration, preventing loops in the trading logic (e.g., BUY → SELL → BUY → SELL).

---

## 1. Overview of Incremental Versions

### V1: Basic Data Retrieval

1. We create a file (src/v1_main.py) that connects to Binance with python-binance.
2. It retrieves account info (balances) and open orders for a specified symbol (e.g., TRUMPUSDC).
3. It logs the results. This script forms the foundation for subsequent versions.

### V2: Automated SELL at +1%

1. In src/v2_main.py, we poll for recent BUY trades and automatically place a SELL at +1% above the fill price.
2. We include a --dry-run option that lets us simulate placing SELL orders without actually hitting Binance.
3. We add an optional 7-period moving average (MA7) log. By default, the script prints MA7 values and indicates if it's UP, DOWN, or FLAT relative to the previous iteration.
4. The script tracks old BUY trades on startup (so it doesn't sell previously held positions).

### V3: Full Integration (BUY→SELL→BUY)

1. In src/v3_main.py, we merge all V2 features:  
   • Polling for new BUY trades to place SELL at +1%.  
   • The ring buffer for logging MA7.  
   • The --dry-run mode.  
2. We add detection of new SELL fills. For any new SELL, the script automatically places a BUY order at -1%.  
3. This closes the loop, allowing both BUY→SELL and SELL→BUY logic in a single script.  
4. As before, we keep track of old SELL trades and triggered trades so we don't place multiple orders for the same fill.

### V4: AI-Driven Logic

1. Replace your fixed ±1% logic with an AI-based strategy module.  
2. Ensure that you still track each automatically placed order with origin = "SYSTEM" so you do not create infinite loops.

---

## 2. Basic State Management to Prevent Loops

Use a small in-memory dictionary or a basic table in a database with a field like `origin`:

• `MANUAL` for user-placed orders.  
• `SYSTEM` for automatically created ones.

Then, for every fill event:

1. Check the `origin`.  
2. If `origin == "SYSTEM"` and it's a SELL fill, do not place a new BUY.  
3. In general, only generate a new order if the fill was triggered by a `MANUAL` order or meets your AI logic requirements.

### Example Pseudocode

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
        # If SELL was placed by system, skip creating a new BUY
        if order_info['origin'] == 'SYSTEM':
            print("System SELL filled. Doing nothing further.")
            return
        else:
            # Possibly place a new BUY if your business logic says so
            # or do nothing if you only want to place SELL in response to manual buys
            pass
    elif side == 'BUY':
        # Usually the logic for placing a SELL at +1% belongs here
        # but you might have already done it in on_buy_placed_manually
        pass
```

---

## 3. Implementation Tips

• Maintain minimal dependencies by using the official Binance SDK.  
• Start with an in-memory dictionary; transition to a table if you need more functionality.  
• Keep logs or console prints of major flow steps for easy debugging.  
• Do not overcomplicate partial fills unless the business logic explicitly requires it.  

---

## 4. Summary

By decoupling "MANUAL" vs. "SYSTEM" orders and carefully responding only to certain fill events, you avoid infinite trading loops. You can then expand functionality step by step in future releases.

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