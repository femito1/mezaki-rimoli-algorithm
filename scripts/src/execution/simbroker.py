"""Simulated broker for backtests.

Executes orders against the latest bar price with optional slippage and
per-order commissions. Extend to add limits/stops or volume constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.lib.types import Order, Fill, OrderSide


@dataclass
class SimBrokerConfig:
    commission_per_order: float = 0.0
    slippage_bps: float = 0.0  # basis points applied to price (one-way)


class SimBroker:
    def __init__(self, config: Optional[SimBrokerConfig] = None) -> None:
        self.config = config or SimBrokerConfig()

    def execute(self, order: Order, last_price: float, timestamp) -> Fill:
        slip = last_price * (self.config.slippage_bps / 10_000.0)
        if order.side == OrderSide.BUY:
            price = last_price + slip
            qty_delta = order.quantity
        else:
            price = last_price - slip
            qty_delta = -order.quantity
        fee = self.config.commission_per_order
        return Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            timestamp=timestamp,
            fee=fee,
        )


