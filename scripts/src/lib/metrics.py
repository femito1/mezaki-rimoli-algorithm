"""Portfolio analytics as pure functions.

Given a return series (preferably log returns), compute common metrics. These
are side-effect free and usable from notebooks, scripts, or the backtester.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_series(returns) -> pd.Series:
    if isinstance(returns, pd.Series):
        return returns.dropna()
    return pd.Series(returns.squeeze()).dropna()


def sharpe(returns, risk_free_annual: float = 0.0, periods_per_year: float = 252.0, log_returns: bool = True, annualize: bool = True) -> float:
    r = ensure_series(returns)
    rf_per = np.log1p(risk_free_annual) / periods_per_year if log_returns else risk_free_annual / periods_per_year
    excess = r - rf_per
    daily = excess.mean() / excess.std()
    return np.sqrt(periods_per_year) * daily if annualize else daily


def sortino(returns, target: float = 0.0, periods_per_year: float = 252.0, annualize: bool = True) -> float:
    r = ensure_series(returns)
    downside = np.minimum(r - target, 0.0)
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return np.nan
    ratio = (r.mean() - target) / downside_dev
    return np.sqrt(periods_per_year) * ratio if annualize else ratio


def omega(returns, threshold: float = 0.0) -> float:
    r = ensure_series(returns)
    gains = np.clip(r - threshold, a_min=0.0, a_max=None).sum()
    losses = np.clip(threshold - r, a_min=0.0, a_max=None).sum()
    return np.inf if losses == 0 else gains / losses


def equity_curve(returns) -> pd.Series:
    r = ensure_series(returns)
    return np.exp(r).cumprod()


def max_drawdown(returns) -> float:
    eq = equity_curve(returns).fillna(0)
    dd = eq / eq.cummax() - 1.0
    return dd.min()


def cagr(returns, periods_per_year: float = 252.0) -> float:
    r = ensure_series(returns)
    eq = np.exp(r).cumprod()
    if len(eq) <= 1:
        return np.nan
    years = len(eq) / periods_per_year
    return eq.iloc[-1] ** (1.0 / years) - 1.0


def calmar(returns, periods_per_year: float = 252.0) -> float:
    c = cagr(returns, periods_per_year=periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return c / mdd


