"""Bridge from our R-based trade lists to Vibe-Trading's validation tools (agent/backtest/validation.py):

  - bootstrap_sharpe_ci: resamples daily returns -> confidence interval and P(Sharpe > 0).
    This is the edge question: could the result be luck?
  - walk_forward_analysis: splits the history into windows -> how consistent across time.
  - monte_carlo_test: reshuffles the order of the same trades. It does NOT test the edge (the same
    trades in another order have the same total); read it as drawdown luck: how often a random
    order would have had a smaller drawdown.

Trades are sized at a fixed 1% of a 10,000 account per 1R (not compounded), so the equity curve is
the R curve in money.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_AGENT = Path(__file__).resolve().parents[3] / "agent"
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

from backtest.models import TradeRecord  # noqa: E402
from backtest.validation import bootstrap_sharpe_ci, monte_carlo_test, walk_forward_analysis  # noqa: E402

CAPITAL = 10_000.0
PER_R = 100.0   # 1% of the starting capital per 1R


def to_records(df: pd.DataFrame) -> list[TradeRecord]:
    out = []
    for row in df.itertuples(index=False):
        out.append(TradeRecord(
            symbol=row.symbol, direction=int(row.direction), entry_price=float(row.entry), exit_price=float(row.exit),
            entry_time=pd.Timestamp(row.entry_time), exit_time=pd.Timestamp(row.exit_time), size=1.0, leverage=1.0,
            pnl=float(row.r_net) * PER_R, pnl_pct=float(row.r_net), exit_reason=str(row.exit_kind),
            holding_bars=float((row.exit_time - row.entry_time).total_seconds() / 3600.0), commission=float(row.costs_r) * PER_R,
        ))
    return out


def daily_equity(df: pd.DataFrame, start: str = "2020-01-01", end: str = "2026-10-01") -> pd.Series:
    days = pd.date_range(start, end, freq="B")
    pnl = df.groupby(df["exit_time"].dt.normalize())["r_net"].sum() * PER_R
    pnl = pnl.reindex(days, fill_value=0.0)
    return CAPITAL + pnl.cumsum()


def validate(df: pd.DataFrame, windows: int = 6) -> dict:
    trades = to_records(df.sort_values("exit_time"))
    eq = daily_equity(df)
    boot = bootstrap_sharpe_ci(eq, n_bootstrap=2000, bars_per_year=252, seed=42, initial_capital=CAPITAL)
    wf = walk_forward_analysis(eq, trades, n_windows=windows, bars_per_year=252)
    mc = monte_carlo_test(trades, CAPITAL, n_simulations=2000, seed=42, bars_per_year=252)
    return {
        "sharpe": boot.get("observed_sharpe"),
        "sharpe_ci95": (boot.get("ci_lower"), boot.get("ci_upper")),
        "p_sharpe_positive": boot.get("prob_positive"),
        "wf_profitable_windows": f"{wf.get('profitable_windows')}/{wf.get('n_windows')}",
        "wf_windows": [(w["start"], w["end"], round(w["return"] * 100, 1)) for w in wf.get("windows", [])],
        "mc_dd_luck_p": mc.get("p_value_max_dd"),
    }
