"""Step 1: does the simulator reproduce the cTrader backtests we already have on gold?

cTrader references (XAUUSD H1, 01/01/2020 - 25/09/2026, 1% risk, the user's broker):
  QTX both directions, ADX 15:  590-597 trades, PF 1.22-1.25, about +57 to +60R, max DD 11-13.5R
  QBX long only:                about +64 to +68R, PF 1.12-1.14, max DD 53-56R (1,166 trades)
"""
from __future__ import annotations

import time

from qlab.costs import load_specs
from qlab.data import load_market
from qlab.models import QbxParams, QtxParams, sim_qbx, sim_qtx, summary, to_frame


def main() -> None:
    specs = load_specs()
    t0 = time.time()
    mk = load_market("XAUUSD", "2020-01-01", "2026-09-26")
    print(f"XAUUSD: {len(mk.m5):,} M5 rows, {len(mk.h1):,} H1 bars ({time.time() - t0:.1f}s)")

    for name, fn, params in [("QTX both, ADX 15", sim_qtx, QtxParams()), ("QBX long only", sim_qbx, QbxParams())]:
        t0 = time.time()
        df = to_frame(fn(mk, specs["XAUUSD"], params))
        s = summary(df)
        g = summary(df, "r_gross")
        by_year = df.groupby(df["entry_time"].dt.year)["r_net"].sum().round(1).to_dict()
        print(f"\n{name}: {s['trades']} trades, net {s['total_r']:+.1f}R (gross {g['total_r']:+.1f}R), PF {s['pf']:.2f}, "
              f"max DD {s['max_dd_r']:.1f}R, win rate {s['win_rate']:.0%}  ({time.time() - t0:.1f}s)")
        print("  by year:", by_year)
        print("  exits:", df["exit_kind"].value_counts().to_dict())


if __name__ == "__main__":
    main()
