# mezaki-rimoli trading algorithm

## Installation

1. Install Python.
2. Install all requirements located in `requirements.txt`.
3. Run `python scripts/example_run.py` from the root of this directory.

## Quick start

The framework wires `DataFeed → Strategy → Broker → Portfolio` through a simple backtest engine.

```python
from src.lib.data import DataFeed, DataFeedConfig
from src.strategies.example import BuyAndHold
from src.backtest.engine import Backtester
from src.execution.simbroker import SimBroker, SimBrokerConfig

data = DataFeed(DataFeedConfig(symbols=["SPY"], start="2000-01-01"))
strat = BuyAndHold(symbol="SPY")
broker = SimBroker(SimBrokerConfig(commission_per_order=0.0, slippage_bps=0.0))
bt = Backtester(data=data, strategy=strat, broker=broker)
result = bt.run()
```

## Data

- `DataFeedConfig` accepts either a `start`/`end` date range or a `period` such as `"max"`, `"10y"`, `"1y"`.
- Example using period:

```python
data = DataFeed(DataFeedConfig(symbols=["SPY"], period="max"))
```

## Strategies

All strategies subclass `src.strategies.base.Strategy` and implement `on_bar`. Some examples included:

- Buy & Hold (`src/strategies/example.py`)

```python
from src.strategies.example import BuyAndHold
strat = BuyAndHold(symbol="SPY")
```

- SMA momentum (`src/strategies/sma.py`): long when price > SMA; volatility targeting and rebalance band.

```python
from src.strategies.sma import SMA
strat = SMA(vol_target_ann=0.15, sma_window=200, vol_window=30, symbol="SPY")
```

- RSI momentum (`src/strategies/momentum_rsi.py`): long when RSI > 50; volatility targeting and rebalance band.

```python
from src.strategies.momentum_rsi import RSI
strat = RSI(vol_target_ann=0.15, alpha_window=14, vol_window=30, symbol="SPY")
```

- RSI mean reversion (`src/strategies/reversion_rsi.py`): long when RSI < 30, short when RSI > 70; symmetric caps.

```python
from src.strategies.reversion_rsi import MR_RSI
strat = MR_RSI(vol_target_ann=0.15, alpha_window=14, vol_window=30, symbol="SPY")
```

## Broker costs (UX)

Broker costs can be configured and passed to the backtester:

```python
from src.execution.simbroker import SimBroker, SimBrokerConfig
broker = SimBroker(SimBrokerConfig(commission_per_order=1.00, slippage_bps=5))
bt = Backtester(data=data, strategy=strat, broker=broker)
```

The example runner `scripts/example_run.py` shows these parameters and prints the trade count.

## Benchmarking vs Buy & Hold

Use `src/backtest/benchmark.py` to compare your strategy against a buy‑and‑hold baseline and plot both equity curves:

```python
from src.backtest.benchmark import compare_to_buy_and_hold
cmp = compare_to_buy_and_hold(bt, symbol="SPY", title="Strategy vs Buy & Hold", show=True)
print(cmp.strategy_metrics)
print(cmp.benchmark_metrics)
```

## Metrics

Common metrics are available in `src/lib/metrics.py`: `sharpe`, `sortino`, `omega`, `cagr`, `calmar`, `max_drawdown`.
