from __future__ import annotations

from typing import List, Optional
import pandas as pd
import numpy as np

from src.strategies.base import Strategy
from src.lib.types import Bar, Order, OrderSide, OrderType

class MR_RSI(Strategy):
    "Long when RSI < 30 (reversion). Short when RSI > 70, flat otherwise"
    def __init__(self, vol_target_ann: float, alpha_window: int, vol_window: int, symbol: str, config: Optional[dict] = None ) -> None:
        super().__init__()
        self.symbol = symbol
        self.alpha_window = alpha_window
        self.vol_window = vol_window
        self.vol_target_ann = vol_target_ann

    def prepare(self, datafeed):
        eps = 1e-6
        df = datafeed.history()
        df = df[df["symbol"] == self.symbol].sort_values("datetime").set_index("datetime")
        r = df["close"].diff()

        df["avg_gains"] = r.clip(lower=0).ewm(
            alpha=1/self.alpha_window, 
            min_periods=self.alpha_window, 
            adjust=False).mean()

        df["avg_losses"] = -r.clip(upper=0).ewm(
            alpha=1/self.alpha_window, 
            min_periods=self.alpha_window, 
            adjust=False).mean()

        df["rs"] = df["avg_gains"] / (df["avg_losses"] + eps)
        df["rsi"] = 100 - 100 / (1 + df["rs"])

        logret = np.log(df["close"]).diff()
        df["vol_ann"] = (logret.rolling(
            self.vol_window, 
            min_periods=self.vol_window)
            .std(ddof=0) * np.sqrt(252)
            ).clip(lower=1e-6)

        self.feat = df[["close", "rsi", "vol_ann"]]

    def on_bar(self, bar: Bar, history: pd.DataFrame, portfolio) -> Optional[List[Order]]:
        
        t = bar.timestamp
        if len(self.feat.loc[:t]) < max(self.vol_window, self.alpha_window) + 1:
            return []
        
        row_t_minus_1 = self.feat.loc[:t].iloc[-2]

        # Extract scalars to avoid Series vs Series comparisons
        rsi_t_1 = float(row_t_minus_1["rsi"].iloc[0])
        vol_ann_t_1 = float(row_t_minus_1["vol_ann"].iloc[0])

        max_w = 2
        vol_target = self.vol_target_ann
        eps = 1e-6

        # If volatility is not finite, skip trading this bar
        if not np.isfinite(vol_ann_t_1):
            return []

        signal_t = 0
        if rsi_t_1 < 30:
            signal_t = 1
        elif rsi_t_1 > 70:
            signal_t = -1
        w_raw_t = vol_target / max(vol_ann_t_1, eps)
        w_t = float(np.clip(signal_t * w_raw_t, -max_w, max_w))

        qty = portfolio.positions.get(self.symbol, 0)
        equity = portfolio.cash + qty * bar.close
        if equity <= 0:
            return []
        a_t = (qty * bar.close) / equity
        # 2% rebalance band
        if abs(w_t - a_t) < 0.02 and a_t != 0:
            return []

        shares_target_t = int(np.floor(w_t * equity / bar.close))
        dq_t = shares_target_t - qty
        if dq_t == 0:
            return []

        affordable = int(portfolio.cash // bar.close)
        max_position = int((equity * max_w) // bar.close)

        if dq_t > 0:
            # Symmetric cap: long side limited by same max_w-based cap
            cap_long_additional = max(0, max_position - qty)
            dq_t = min(dq_t, affordable, cap_long_additional)
        if dq_t < 0:
            dq_t = max(dq_t, -max_position - qty)
        
        side = OrderSide.BUY if dq_t > 0 else OrderSide.SELL
        return [Order(symbol=self.symbol, side=side, quantity=abs(dq_t), type=OrderType.MARKET)]
