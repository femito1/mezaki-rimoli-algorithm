"""Backtesting event loop.

Connects DataFeed → Strategy → Broker → Portfolio on a bar-by-bar basis.
Returns equity/returns and basic stats in a `BacktestResult`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from src.lib.data import DataFeed
from src.lib.portfolio import Portfolio
from src.lib.types import Bar, Order
from src.execution.simbroker import SimBroker, SimBrokerConfig


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: int


class Backtester:
    def __init__(self, data: DataFeed, strategy, portfolio: Optional[Portfolio] = None, broker: Optional[SimBroker] = None) -> None:
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio or Portfolio()
        self.broker = broker or SimBroker(SimBrokerConfig())

    def run(self) -> BacktestResult:
        hist = self.data.history()
        self.strategy.prepare(self.data)

        trades = 0
        last_prices: Dict[str, float] = {}

        for bar in self.data.iter_bars():
            last_prices[bar.symbol] = bar.close

            orders = self.strategy.on_bar(bar, hist, self.portfolio) or []
            for order in orders:
                fill = self.broker.execute(order, last_price=bar.close, timestamp=bar.timestamp)
                qty_delta = fill.quantity if fill.side.value == 1 else -fill.quantity
                self.portfolio.apply_fill(fill.symbol, qty_delta, fill.price, fee=fill.fee)
                self.strategy.on_fill(fill, self.portfolio)
                trades += 1

            self.portfolio.update_mark_to_market(last_prices)

        # allow strategy to finalize and compute a report
        self.strategy.finalize(self.portfolio)

        return BacktestResult(
            equity=self.portfolio.equity_series(),
            returns=self.portfolio.returns_series(),
            trades=trades,
        )


