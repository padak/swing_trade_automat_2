# Trend Detector v3 Documentation

## Overview
The Trend Detector v3 is an automated trading system that uses technical analysis indicators and machine learning to detect market trends and execute trades on the Binance exchange. The system is specifically designed to trade the TRUMP/USDC pair with enhanced risk management and multiple confirmation signals.

## System Architecture

### Sequence Diagram
```mermaid
sequenceDiagram
    participant Main
    participant Binance
    participant Strategy
    participant Logger
    
    Main->>Main: Initialize parameters
    Main->>Binance: Initialize client
    Main->>Binance: Fetch initial price
    Main->>Main: Calculate initial balances
    
    loop Trading Loop
        Main->>Binance: Fetch current market data
        Binance-->>Main: Return price & volume
        Main->>Strategy: Generate trading signals
        Strategy->>Strategy: Calculate indicators (MA, RSI, MACD)
        Strategy->>Strategy: Check trend strength
        Strategy->>Strategy: Check volume ratio
        Strategy-->>Main: Return signal & trend
        
        alt Signal is BUY or SELL
            Main->>Strategy: Execute trade
            Strategy->>Strategy: Check minimum hold time
            Strategy->>Strategy: Validate trade conditions
            Strategy-->>Main: Return trade details
            Main->>Logger: Log trade execution
        else Signal is NEUTRAL
            Main->>Logger: Log market state
        end
        
        Main->>Main: Update portfolio value
        Main->>Logger: Log performance metrics
        Main->>Main: Wait for next iteration
    end
```

### Flow Chart
```mermaid
flowchart TD
    A[Start] --> B[Initialize Client]
    B --> C[Load Strategy Parameters]
    C --> D[Fetch Initial Price]
    D --> E[Calculate Initial Portfolio]
    E --> F[Enter Trading Loop]
    
    F --> G[Fetch Market Data]
    G --> H[Update Price History]
    
    H --> I{Enough Historical Data?}
    I -- No --> W[Wait for More Data]
    W --> F
    
    I -- Yes --> J[Generate Trading Signal]
    J --> K[Calculate Technical Indicators]
    K --> L[Check Trend Strength]
    L --> M[Check Volume Ratio]
    
    M --> N{Signal Type?}
    N -- BUY --> O[Check Buy Conditions]
    N -- SELL --> P[Check Sell Conditions]
    N -- NEUTRAL --> Q[Log Current State]
    
    O --> R{Minimum Hold Time Met?}
    P --> R
    
    R -- Yes --> S[Execute Trade]
    R -- No --> Q
    
    S --> T[Update Balances]
    T --> U[Log Trade Details]
    U --> Q
    
    Q --> V[Calculate Performance]
    V --> F
```

## Key Components

1. **Initialization**
   - Load API credentials
   - Set initial strategy parameters
   - Calculate initial portfolio values

2. **Market Data Collection**
   - Fetch real-time price and volume data
   - Maintain historical price records
   - Calculate volume ratios

3. **Technical Analysis**
   - Moving Averages (Fast & Slow)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Trend Strength Indicators

4. **Trading Logic**
   - Signal generation based on multiple confirmations
   - Minimum hold time enforcement
   - Volume threshold validation
   - Trend strength confirmation

5. **Risk Management**
   - Stop Loss
   - Take Profit
   - Trade Risk Percentage
   - Minimum Trade Size

6. **Performance Tracking**
   - Portfolio value calculation
   - Profit/Loss tracking
   - Trade history logging
   - Performance metrics calculation

## Strategy Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| FAST_MA_PERIOD | 10 | Fast Moving Average period |
| SLOW_MA_PERIOD | 30 | Slow Moving Average period |
| RSI_PERIOD | 14 | RSI calculation period |
| RSI_OVERBOUGHT | 65 | RSI overbought threshold |
| RSI_OVERSOLD | 35 | RSI oversold threshold |
| MIN_HOLD_TIME_MINUTES | 1 | Minimum time between trades |
| VOLUME_THRESHOLD | 1.1 | Minimum volume multiplier |
| TREND_STRENGTH_THRESHOLD | 0.3 | Required trend strength |

## Trading Signals

The system generates trading signals based on the following conditions:

### Buy Signal
- Price is in downtrend
- RSI is oversold OR trend reversal detected
- Volume above threshold
- Strong trend confirmation
- Significant price movement
- MACD bullish confirmation

### Sell Signal
- Price is in uptrend
- RSI is overbought OR trend reversal detected
- Volume above threshold
- Strong trend confirmation
- Significant price movement
- MACD bearish confirmation

## Logging and Monitoring

The system maintains detailed logs of:
- Trade executions
- Portfolio performance
- Market conditions
- Technical indicators
- Trade statistics

## Error Handling

The system includes:
- Graceful shutdown handling
- API connection error recovery
- Market data validation
- Trade execution verification
- Balance update confirmation
