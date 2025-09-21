"""Example strategy implementations.

`BuyAndHold` demonstrates the minimal surface area: buy on first bar, hold.
Use this file as a template to build more complex strategies.
"""
from __future__ import annotations

from typing import List, Optional
import pandas as pd

from src.strategies.base import Strategy
from src.lib.types import Bar, Order, OrderSide, OrderType


class BuyAndHold(Strategy):
    """Buys one share on the first bar and holds."""

    def __init__(self, symbol: str, shares: int = 1, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.symbol = symbol
        self.shares = shares
        self.has_position = False

    def on_bar(self, bar: Bar, history: pd.DataFrame, portfolio) -> Optional[List[Order]]:
        if not self.has_position and bar.symbol == self.symbol:
            self.has_position = True
            return [Order(symbol=self.symbol, side=OrderSide.BUY, quantity=(portfolio.cash / bar.close), type=OrderType.MARKET)]
        return []


