"""Simple portfolio accounting.

Tracks cash, integer positions, equity curve, and log returns. The backtester
updates mark-to-market each bar and applies fills to update cash/positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Portfolio:
    cash: float = 100_000.0
    positions: Dict[str, int] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def update_mark_to_market(self, prices: Dict[str, float]) -> None:
        equity = self.cash + sum(qty * prices.get(sym, 0.0) for sym, qty in self.positions.items())
        prev = self.equity_curve[-1] if self.equity_curve else equity
        self.equity_curve.append(equity)
        self.returns.append(0.0 if prev == 0 else np.log(equity / prev))

    def apply_fill(self, symbol: str, quantity_delta: int, price: float, fee: float = 0.0) -> None:
        self.cash -= quantity_delta * price + fee
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity_delta

    def equity_series(self) -> pd.Series:
        return pd.Series(self.equity_curve)

    def returns_series(self) -> pd.Series:
        return pd.Series(self.returns)


