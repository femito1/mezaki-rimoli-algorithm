"""Example: run a simple Buy & Hold backtest and print metrics."""
from __future__ import annotations

import numpy as np

from src.lib.data import DataFeed, DataFeedConfig
from src.strategies.example import BuyAndHold
from src.backtest.engine import Backtester
from src.lib.metrics import sharpe, sortino, omega, cagr, calmar, max_drawdown


def main():
    data = DataFeed(DataFeedConfig(symbols=["SPY"], start="2000-01-01"))
    df = data.history()
    print(df.head())
    strat = BuyAndHold(symbol="SPY", shares=1)

    bt = Backtester(data=data, strategy=strat)
    result = bt.run()

    r = result.returns.replace([np.inf, -np.inf], np.nan).dropna()
    print("Sharpe:", sharpe(r))
    print("Sortino:", sortino(r))
    print("Omega:", omega(r))
    print("CAGR:", cagr(r))
    print("Calmar:", calmar(r))
    print("Max Drawdown:", max_drawdown(r))


if __name__ == "__main__":
    main()


