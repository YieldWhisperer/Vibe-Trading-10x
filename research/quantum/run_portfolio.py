"""Step 3: combine the market/model streams that passed step 2 into one portfolio, and compare it
with gold QTX alone (the only strategy that has passed every test so far).

Each stream risks the same 1% per trade (not compounded); the portfolio is the sum of the R streams.
Validation of every candidate goes through Vibe-Trading's tools (qlab/validate.py).

Usage: python run_portfolio.py                 # every PASS row in results_markets.csv + gold QTX
       python run_portfolio.py XAUUSD:QTX US500:QTX ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.data import CACHE_DIR
from qlab.models import summary
from qlab.validate import validate

SPLIT = pd.Timestamp("2024-01-01")
HERE = Path(__file__).parent


def load(stream: str) -> pd.DataFrame:
    sym, model = stream.split(":")
    df = pd.read_pickle(CACHE_DIR / "trades" / f"{sym}_{model}.pkl")
    df["stream"] = stream
    return df


def report(name: str, df: pd.DataFrame) -> dict:
    df = df.sort_values("exit_time")
    s, a, b = summary(df), summary(df[df.entry_time < SPLIT]), summary(df[df.entry_time >= SPLIT])
    years = df.groupby(df["exit_time"].dt.year)["r_net"].sum()
    v = validate(df)
    row = {
        "portfolio": name, "trades": s["trades"], "net_r": round(s["total_r"], 1), "pf": round(s["pf"], 2),
        "max_dd_r": round(s["max_dd_r"], 1), "ret_dd": round(s["total_r"] / s["max_dd_r"], 2) if s["max_dd_r"] else np.nan,
        "A_r": round(a["total_r"], 1), "A_pf": round(a["pf"], 2), "B_r": round(b["total_r"], 1), "B_pf": round(b["pf"], 2),
        "losing_years": ", ".join(f"{y} {r:+.1f}R" for y, r in years.items() if r < 0) or "none",
        "sharpe": v["sharpe"], "sharpe_ci95": v["sharpe_ci95"], "p_sharpe>0": v["p_sharpe_positive"],
        "walk_forward": v["wf_profitable_windows"],
    }
    return row


def main(streams: list[str]) -> None:
    if not streams:
        table = pd.read_csv(HERE / "results_markets.csv")
        streams = [f"{r.symbol}:{r.model}" for r in table.itertuples() if r.PASS == "PASS"]
        if "XAUUSD:QTX" not in streams:
            streams.insert(0, "XAUUSD:QTX")
    data = {s: load(s) for s in streams}

    print("Streams:", ", ".join(streams))
    monthly = pd.DataFrame({s: d.groupby(d["exit_time"].dt.to_period("M"))["r_net"].sum() for s, d in data.items()}).fillna(0.0)
    print("\nMonthly R correlation:")
    print(monthly.corr().round(2).to_string())

    rows = [report(s, d) for s, d in data.items()]
    rows.append(report("PORTFOLIO (all above)", pd.concat(data.values(), ignore_index=True)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results_portfolio.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
