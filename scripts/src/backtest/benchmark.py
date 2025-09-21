"""Benchmark utilities: compare a strategy to buy-and-hold for a symbol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.engine import Backtester, BacktestResult
from src.lib.metrics import sharpe, sortino, omega, cagr, calmar, max_drawdown


@dataclass
class Comparison:
    strategy_equity: pd.Series
    benchmark_equity: pd.Series
    strategy_metrics: dict
    benchmark_metrics: dict


def _symbol_history_df(backtester: Backtester, symbol: str) -> pd.DataFrame:
    df = backtester.data.history()
    df = df[df["symbol"] == symbol].sort_values("datetime").reset_index(drop=True)
    return df


def _buy_and_hold_equity(
    df_symbol: pd.DataFrame,
    initial_equity: float,
    commission_per_order: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.Series:
    """Construct a buy-and-hold equity curve with optional one-time entry costs.

    If costs are zero, scales equity by close/base. If costs are non-zero, simulate
    buying integer shares at an entry price that includes slippage and subtract
    commission; keep leftover cash in the equity path.
    """
    close = df_symbol["close"].astype(float).reset_index(drop=True)
    base = float(close.iloc[0])

    if commission_per_order == 0.0 and slippage_bps == 0.0:
        equity = initial_equity * (close / base)
        equity.index = pd.to_datetime(df_symbol["datetime"])  # set datetime index
        return equity

    entry_price = base * (1.0 + slippage_bps / 10_000.0)
    cash_after_commission = max(0.0, initial_equity - commission_per_order)
    shares = int(np.floor(cash_after_commission / entry_price))
    leftover_cash = cash_after_commission - shares * entry_price
    equity = shares * close + leftover_cash
    equity.index = pd.to_datetime(df_symbol["datetime"])  # set datetime index
    return equity


def _attach_index(equity: pd.Series, index: pd.Index) -> pd.Series:
    eq = equity.copy()
    eq.index = index[: len(eq)]  # truncate if needed
    return eq


def compare_to_buy_and_hold(
    backtester: Backtester,
    symbol: str,
    title: Optional[str] = None,
    show: bool = True,
    benchmark_commission: float = 0.0,
    benchmark_slippage_bps: float = 0.0,
) -> Comparison:
    """Run backtest, build a buy-and-hold baseline, compute metrics, and optionally plot.

    Assumes the DataFeed contains the symbol and the strategy operates on that symbol/time range.
    """
    result: BacktestResult = backtester.run()

    # Build symbol history and benchmark equity
    sym_df = _symbol_history_df(backtester, symbol)
    if sym_df.empty or len(result.equity) == 0:
        raise ValueError("No data or empty backtest result to compare.")

    # Align indices
    dt_index = pd.to_datetime(sym_df["datetime"])  # expected length ~ number of bars for symbol
    strat_equity = _attach_index(result.equity, dt_index)
    initial_eq = float(strat_equity.iloc[0])
    bh_equity = _buy_and_hold_equity(
        sym_df,
        initial_equity=initial_eq,
        commission_per_order=benchmark_commission,
        slippage_bps=benchmark_slippage_bps,
    )

    # Compute returns (log) from equity for both
    def eq_to_log_returns(eq: pd.Series) -> pd.Series:
        eq = eq.astype(float)
        r = np.log(eq).diff().replace([np.inf, -np.inf], np.nan).dropna()
        return r

    r_strat = eq_to_log_returns(strat_equity)
    r_bh = eq_to_log_returns(bh_equity)

    strat_metrics = {
        "sharpe": float(sharpe(r_strat)),
        "sortino": float(sortino(r_strat)),
        "omega": float(omega(r_strat)),
        "cagr": float(cagr(r_strat)),
        "calmar": float(calmar(r_strat)),
        "max_drawdown": float(max_drawdown(r_strat)),
    }
    bh_metrics = {
        "sharpe": float(sharpe(r_bh)),
        "sortino": float(sortino(r_bh)),
        "omega": float(omega(r_bh)),
        "cagr": float(cagr(r_bh)),
        "calmar": float(calmar(r_bh)),
        "max_drawdown": float(max_drawdown(r_bh)),
    }

    if show:
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(strat_equity.index, strat_equity.values, label="Strategy")
        ax.plot(bh_equity.index, bh_equity.values, label=f"Buy & Hold {symbol}")
        ax.set_title(title or f"Strategy vs Buy & Hold ({symbol})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    return Comparison(
        strategy_equity=strat_equity,
        benchmark_equity=bh_equity,
        strategy_metrics=strat_metrics,
        benchmark_metrics=bh_metrics,
    )


