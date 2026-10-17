"""Step 2: QTX and QBX, default settings, on every market with broker data.

Pre-registered pass rule (fixed before any result was seen, 2026-10-17):
  period A 2020-01-01..2023-12-31 (discovery) and period B 2024-01-01..2026-09-30 (held out);
  pass = PF >= 1.15 in A and PF >= 1.10 in B, with >= 50 trades in A and >= 30 in B.
No per-market tuning: the same settings everywhere.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from qlab.costs import load_specs
from qlab.data import CACHE_DIR, available_symbols, load_market
from qlab.models import QbxParams, QtxParams, sim_qbx, sim_qtx, summary, to_frame

SPLIT = pd.Timestamp("2024-01-01")
END = "2026-10-01"


def evaluate(df: pd.DataFrame) -> dict:
    a = df[df.entry_time < SPLIT]
    b = df[df.entry_time >= SPLIT]
    sa, sb, s = summary(a), summary(b), summary(df)
    passed = sa["pf"] >= 1.15 and sb["pf"] >= 1.10 and sa["trades"] >= 50 and sb["trades"] >= 30
    return {"trades": s["trades"], "net_r": round(s["total_r"], 1), "pf": round(s["pf"], 2), "dd_r": round(s["max_dd_r"], 1),
            "A_trades": sa["trades"], "A_pf": round(sa["pf"], 2), "A_r": round(sa["total_r"], 1),
            "B_trades": sb["trades"], "B_pf": round(sb["pf"], 2), "B_r": round(sb["total_r"], 1),
            "PASS": "PASS" if passed else ""}


def main(symbols: list[str]) -> None:
    specs = load_specs()
    rows = []
    out = CACHE_DIR / "trades"
    out.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        if sym not in specs:
            print(f"{sym}: no cost spec - skipped")
            continue
        t0 = time.time()
        mk = load_market(sym, "2020-01-01", END)
        for name, fn, p in [("QTX", sim_qtx, QtxParams()), ("QBX", sim_qbx, QbxParams())]:
            df = to_frame(fn(mk, specs[sym], p))
            df.to_pickle(out / f"{sym}_{name}.pkl")
            row = {"symbol": sym, "model": name, **evaluate(df)}
            rows.append(row)
            print(row, flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(Path(__file__).with_name("results_markets.csv"), index=False)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or available_symbols())
