"""Example: run a simple Buy & Hold backtest and print metrics."""
from __future__ import annotations

import numpy as np

from src.lib.data import DataFeed, DataFeedConfig
from src.strategies.example import BuyAndHold
from src.backtest.engine import Backtester
from src.lib.metrics import sharpe, sortino, omega, cagr, calmar, max_drawdown
from src.execution.simbroker import SimBroker, SimBrokerConfig


def main():
    data = DataFeed(DataFeedConfig(symbols=["SPY"], start="2000-01-01"))
    df = data.history()
    print(df.head())
    strat = BuyAndHold(symbol="SPY", shares=1)

    # Broker UX: configure commission and slippage in one place
    broker_cfg = SimBrokerConfig(
        commission_per_order=0.0,  # set e.g. 1.00 for $1 commission per order
        slippage_bps=0.0,          # set e.g. 5 for 5 bps slippage one-way
    )
    broker = SimBroker(broker_cfg)

    bt = Backtester(data=data, strategy=strat, broker=broker)
    result = bt.run()

    r = result.returns.replace([np.inf, -np.inf], np.nan).dropna()
    print("Trades:", result.trades)
    print("Sharpe:", sharpe(r))
    print("Sortino:", sortino(r))
    print("Omega:", omega(r))
    print("CAGR:", cagr(r))
    print("Calmar:", calmar(r))
    print("Max Drawdown:", max_drawdown(r))


if __name__ == "__main__":
    main()


