"""Core typed primitives used across the framework.

Defines light-weight data structures for bars, orders, and fills, as well as
basic enums. These keep modules decoupled and make strategies easy to write.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class OrderSide(Enum):
    BUY = 1
    SELL = -1


class OrderType(Enum):
    MARKET = "market"


@dataclass
class Bar:
    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    type: OrderType = OrderType.MARKET


@dataclass
class Fill:
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: pd.Timestamp
    fee: float = 0.0


