"""Step 2b: are the streams that passed real, or luck / plain market drift?

For each candidate stream:
  1. Random-entry benchmark: the same model's exits and management, but entries at random H1 bars
     (QTX keeps its long/short mix, QBX stays long). If random entries do as well, the "edge" is the
     market's drift (e.g. a bull run), not the signal.  p = share of random runs with a mean R per
     trade at least as good as the real one.
  2. Cost stress: spreads and stop slippage doubled, and no swap credits (positive swaps set to 0).
  3. Neighbouring settings: one setting at a time moved to each side; a real edge should not depend
     on the exact defaults.
"""
from __future__ import annotations

import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.costs import load_specs
from qlab.data import load_market
from qlab.models import QbxParams, QtxParams, sim_qbx, sim_qtx, summary, to_frame

SPLIT = pd.Timestamp("2024-01-01")
START, END = "2020-01-01", "2026-10-01"
RANDOM_RUNS = 40
HERE = Path(__file__).parent

QTX_VARIANTS = {"channel 15": {"channel": 15}, "channel 25": {"channel": 25}, "ADX 12": {"adx_min": 12.0}, "ADX 18": {"adx_min": 18.0},
                "stop 2.0 ATR": {"stop_atr": 2.0}, "stop 3.0 ATR": {"stop_atr": 3.0}, "BE 0.5R": {"break_even_r": 0.5}, "BE 1.0R": {"break_even_r": 1.0}}
QBX_VARIANTS = {"lookback 15": {"lookback": 15}, "lookback 30": {"lookback": 30}, "stop 1.5 ATR": {"stop_atr": 1.5}, "stop 2.5 ATR": {"stop_atr": 2.5},
                "target 3R": {"target_r": 3.0}, "target 5R": {"target_r": 5.0}, "BE 0.75R": {"break_even_r": 0.75}, "BE 1.5R": {"break_even_r": 1.5}}


def _run(sym: str, model: str, params=None, entries=None, stress: bool = False) -> pd.DataFrame:
    spec = copy.copy(load_specs()[sym])
    mk = load_market(sym, START, END)
    if stress:
        spec.spread *= 2
        spec.slip *= 2
        spec.swap_long = min(spec.swap_long, 0.0)
        spec.swap_short = min(spec.swap_short, 0.0)
        mk.m5 = mk.m5.copy()
        mk.m5["spread"] = mk.m5["spread"] * 2
    if model == "QTX":
        return to_frame(sim_qtx(mk, spec, params or QtxParams(), entries))
    return to_frame(sim_qbx(mk, spec, params or QbxParams(), entries))


def _random_job(args):
    sym, model, seed, n_trades, directions = args
    mk = load_market(sym, START, END)
    rng = np.random.default_rng(seed)
    bars = np.sort(rng.choice(np.arange(110, len(mk.h1)), size=n_trades, replace=False))
    dirs = rng.permutation(directions)
    df = _run(sym, model, entries={int(b): int(d) for b, d in zip(bars, dirs)})
    return float(df["r_net"].mean()) if len(df) else 0.0


def _variant_job(args):
    sym, model, label, change = args
    params = replace(QtxParams() if model == "QTX" else QbxParams(), **change)
    df = _run(sym, model, params)
    a, b = summary(df[df.entry_time < SPLIT]), summary(df[df.entry_time >= SPLIT])
    return {"stream": f"{sym}:{model}", "test": label, "net_r": round(df["r_net"].sum(), 1),
            "A_pf": round(a["pf"], 2), "B_pf": round(b["pf"], 2)}


def main(streams: list[str]) -> None:
    workers = max(1, (os.cpu_count() or 2) - 1)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for stream in streams:
            sym, model = stream.split(":")
            base = _run(sym, model)
            mean_r = float(base["r_net"].mean())
            dirs = base["direction"].to_numpy()

            jobs = [(sym, model, 1000 + k, len(base), dirs) for k in range(RANDOM_RUNS)]
            rand = np.array(list(pool.map(_random_job, jobs)))
            p_random = float((rand >= mean_r).mean())

            stressed = _run(sym, model, stress=True)
            sa, sb = summary(stressed[stressed.entry_time < SPLIT]), summary(stressed[stressed.entry_time >= SPLIT])

            variants = list(pool.map(_variant_job, [(sym, model, k, v) for k, v in (QTX_VARIANTS if model == "QTX" else QBX_VARIANTS).items()]))
            both_ok = sum(1 for v in variants if v["A_pf"] >= 1.0 and v["B_pf"] >= 1.0)

            row = {"stream": stream, "mean_r": round(mean_r, 3), "random_mean_r": round(float(rand.mean()), 3),
                   "random_p95": round(float(np.percentile(rand, 95)), 3), "p_random": p_random,
                   "stress_net_r": round(stressed["r_net"].sum(), 1), "stress_A_pf": round(sa["pf"], 2), "stress_B_pf": round(sb["pf"], 2),
                   "variants_profitable_both": f"{both_ok}/{len(variants)}"}
            rows.append(row)
            print(row, flush=True)
            for v in variants:
                print("   ", v, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results_robustness.csv", index=False)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["XAUUSD:QTX", "XAGUSD:QTX", "USDJPY:QTX", "USDJPY:QBX", "USTEC:QBX"])
