from __future__ import annotations

from typing import List, Optional
import pandas as pd
import numpy as np

from src.strategies.base import Strategy
from src.lib.types import Bar, Order, OrderSide, OrderType

class SMA(Strategy):
    "Goes long when the price is above the 200-day moving average"

    def __init__(self, vol_target_ann: float, sma_window: int, vol_window: int, symbol: str, shares: int = 1, config: Optional[dict] = None ) -> None:
        super().__init__()
        self.symbol = symbol
        self.shares = shares
        self.vol_window = vol_window
        self.sma_window = sma_window
        self.vol_target_ann = vol_target_ann
        self.has_position = False

    def prepare(self, datafeed):
        df = datafeed.history()
        df = df[df["symbol"] == self.symbol].sort_values("datetime").set_index("datetime")
        df["sma"] = df["close"].rolling(self.sma_window).mean()
        r = np.log(df["close"]).diff()
        df["vol_ann"] = r.rolling(self.vol_window).std() * np.sqrt(252)
        self.feat = df[["close", "sma", "vol_ann"]]

    def on_bar(self, bar: Bar, history: pd.DataFrame, portfolio) -> Optional[List[Order]]:
        
        t = bar.timestamp
        if len(self.feat.loc[:t]) < max(self.sma_window, self.vol_window) + 1:
            return []
        
        row_t_minus_1 = self.feat.loc[:t].iloc[-2]

        # Extract scalars to avoid Series vs Series comparisons
        close_t_1 = float(row_t_minus_1["close"].iloc[0])
        sma_t_1 = float(row_t_minus_1["sma"].iloc[0])
        vol_ann_t_1 = float(row_t_minus_1["vol_ann"].iloc[0])

        max_w = 2
        vol_target = self.vol_target_ann
        eps = 1e-6

        # If volatility is not finite, skip trading this bar
        if not np.isfinite(vol_ann_t_1):
            return []

        signal_t = 1 if close_t_1 > sma_t_1 else 0
        w_raw_t = vol_target / max(vol_ann_t_1, eps)
        w_t = min(max_w, max(0.0, signal_t * w_raw_t))

        qty = portfolio.positions.get(self.symbol, 0)
        equity = portfolio.cash + qty * bar.close
        if equity <= 0:
            return []
        a_t = (qty * bar.close) / equity
        # 2% rebalance band
        if abs(w_t - a_t) < 0.02:
            return []

        shares_target_t = int(np.floor(w_t * equity / bar.close))
        dq_t = shares_target_t - qty
        if dq_t == 0:
            return []
        if dq_t > 0:
            affordable = int(portfolio.cash // bar.close)
            dq_t = min(dq_t, affordable)
        
        side = OrderSide.BUY if dq_t > 0 else OrderSide.SELL
        return [Order(symbol=self.symbol, side=side, quantity=abs(dq_t), type=OrderType.MARKET)]
