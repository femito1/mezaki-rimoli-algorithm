"""Normalized data access layer.

DataFeed returns an iterator of Bar objects and a full history DataFrame in a
long-form schema (datetime, symbol, open, high, low, close). It can either
ingest a user-provided DataFrame or fetch from yfinance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from .types import Bar


@dataclass
class DataFeedConfig:
    symbols: List[str]
    frequency: str = "1D"
    start: Optional[str] = None
    end: Optional[str] = None
    period: Optional[str] = None  # e.g., "max", "10y", "1y"


class DataFeed:
    """Unified datafeed for backtests.

    Accepts either a preloaded DataFrame (long-form with columns: datetime, symbol, open, high, low, close)
    or downloads via yfinance when given only symbols.
    """

    def __init__(self, config: DataFeedConfig, data: Optional[pd.DataFrame] = None) -> None:
        self.config = config
        if data is None:
            import yfinance as yf

            frames = []
            for sym in config.symbols:
                dl_kwargs = dict(group_by="column", auto_adjust=False)
                if config.period:
                    df = yf.download(sym, period=config.period, **dl_kwargs)
                else:
                    df = yf.download(sym, start=config.start, end=config.end, **dl_kwargs)
                df = df.reset_index().rename(
                    columns={
                        "Date": "datetime",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                    }
                )
                df["symbol"] = sym
                frames.append(df[["datetime", "symbol", "open", "high", "low", "close"]])
            data = pd.concat(frames, ignore_index=True)

        # Normalize schema
        required = {"datetime", "symbol", "open", "high", "low", "close"}
        # missing = required - set(map(str.lower, data.columns))
        # Assume user passed correct column names; for simplicity we skip dynamic mapping here
        self.df = data.copy()
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])  # type: ignore[index]
        self.df = self.df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    def iter_bars(self) -> Iterable[Bar]:
        # Use positional tuple access to avoid attribute-name quirks from itertuples
        cols = ["datetime", "symbol", "open", "high", "low", "close"]
        for dt, sym, o, h, l, c in self.df[cols].itertuples(index=False, name=None):
            yield Bar(
                timestamp=pd.Timestamp(dt),
                symbol=str(sym),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
            )

    def history(self) -> pd.DataFrame:
        return self.df.copy()


