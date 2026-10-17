"""Step 2c: how much of the 'pass' is selection luck?

Pick streams using 2020-2023 ONLY (PF >= 1.15, >= 50 trades), then score that same pick on
2024-Sep 2026, which played no part in choosing it. This is the honest out-of-sample estimate of
'choose the markets that worked'."""
import pandas as pd

from qlab.data import CACHE_DIR
from qlab.models import summary

SPLIT = pd.Timestamp("2024-01-01")
t = pd.read_csv("results_markets.csv")
picked = t[(t.A_pf >= 1.15) & (t.A_trades >= 50)]
frames = []
for r in picked.itertuples():
    df = pd.read_pickle(CACHE_DIR / "trades" / f"{r.symbol}_{r.model}.pkl")
    frames.append(df[df.entry_time >= SPLIT])
    print(f"  {r.symbol}:{r.model:4} 2020-23 PF {r.A_pf:.2f} -> 2024-26 PF {r.B_pf:.2f}, {r.B_r:+.1f}R")
b = summary(pd.concat(frames).sort_values("exit_time"))
print(f"Picked on 2020-23: {len(picked)} streams. In 2024-26 together: {b['trades']} trades, {b['total_r']:+.1f}R, PF {b['pf']:.2f}, max DD {b['max_dd_r']:.1f}R")
