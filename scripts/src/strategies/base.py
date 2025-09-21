"""Strategy interface.

Subclass `Strategy` and implement `on_bar` to create new strategies. Optional
hooks: `prepare` (precompute indicators), `on_fill` (react to fills), and
`finalize` (cleanup/reporting after run).

This base class also provides default trade tracking using FIFO lots for
long-only flows, and a `finalize` implementation that computes a summary and
portfolio metrics from the run. Strategies may override any part as needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.lib.types import Bar, Order, Fill
import numpy as np
import pandas as pd
from src.lib import metrics as M


class Strategy(ABC):
    """Base Strategy interface.

    Implementors should keep strategies free of I/O and plotting, and focus on
    signal generation and portfolio-aware decisions.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        # Trade tracking (FIFO lots, long-only by default)
        self._open_lots: dict[str, list[dict]] = {}
        self.trade_log: list[dict] = []
        self.summary: dict = {}
        self.metrics: dict = {}

    def prepare(self, data) -> None:
        """Optional hook called once before the backtest starts.

        Use to precompute indicators/features from the datafeed if needed.
        """
        return None

    @abstractmethod
    def on_bar(self, bar: Bar, history, portfolio) -> Optional[List[Order]]:
        """Called on each new bar.

        Return a list of orders to submit for this bar, or None/[] for no action.
        """
        raise NotImplementedError

    def on_fill(self, fill: Fill, portfolio) -> None:
        """Default trade tracking with FIFO lot matching (long-only).

        Records completed round-trips with realized PnL and holding period.
        Strategies can override to implement custom behavior.
        """
        sym = fill.symbol
        lots = self._open_lots.setdefault(sym, [])
        ts = pd.Timestamp(fill.timestamp)

        if fill.side.name == "BUY":
            lots.append({"qty": fill.quantity, "price": fill.price, "ts": ts})
            return None

        # SELL: match against FIFO longs
        qty_to_close = fill.quantity
        while qty_to_close > 0 and lots:
            lot = lots[0]
            take = min(qty_to_close, lot["qty"])
            realized = (fill.price - lot["price"]) * take
            holding_days = (ts - lot["ts"]).days
            self.trade_log.append({
                "symbol": sym,
                "entry_ts": lot["ts"],
                "exit_ts": ts,
                "entry_price": lot["price"],
                "exit_price": fill.price,
                "qty": take,
                "pnl": realized,
                "ret": (fill.price / lot["price"]) - 1.0,
                "holding_days": holding_days,
            })
            lot["qty"] -= take
            qty_to_close -= take
            if lot["qty"] == 0:
                lots.pop(0)
        return None

    def finalize(self, portfolio) -> None:
        """Compute a summary and metrics from the run.

        Populates `self.summary` and `self.metrics` and `self.report` (combined).
        """
        # Trade-level summary
        trades = self.trade_log
        num_trades = len(trades)
        wins = [t for t in trades if t.get("pnl", 0.0) > 0]
        losses = [t for t in trades if t.get("pnl", 0.0) < 0]
        self.summary = {
            "num_trades": num_trades,
            "win_rate": (len(wins) / num_trades) if num_trades else 0.0,
            "avg_win": float(np.mean([t["pnl"] for t in wins])) if wins else 0.0,
            "avg_loss": float(np.mean([t["pnl"] for t in losses])) if losses else 0.0,
            "avg_holding_days": float(np.mean([t["holding_days"] for t in trades])) if trades else 0.0,
        }

        # Portfolio-level metrics (use log returns from portfolio)
        r = portfolio.returns_series().replace([np.inf, -np.inf], np.nan).dropna()
        self.metrics = {
            "sharpe": float(M.sharpe(r)),
            "sortino": float(M.sortino(r)),
            "omega": float(M.omega(r)),
            "cagr": float(M.cagr(r)),
            "calmar": float(M.calmar(r)),
            "max_drawdown": float(M.max_drawdown(r)),
        }

        self.report = {"summary": self.summary, "metrics": self.metrics}
        return None


