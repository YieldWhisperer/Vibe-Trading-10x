"""Trading costs per market, in price units per unit traded.

Commission and swaps come from the broker's own symbol sheet (Documents\\QuantumData\\symbols.csv,
written by QuantumDataExport, IC Markets / Raw Trading cTrader, captured 2026-10-02) where the
symbol is in it. Like a cTrader backtest, today's swap rates are applied to the whole history.

Spreads: the server history carries no spread, so each market uses a typical raw-account spread
(gold uses the spread recorded in its export). Stop fills (stop-loss exits and breakout stop
entries) pay an extra slippage. Majors not in the sheet use approximate swaps - marked below.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .data import DATA_DIR


@dataclass
class Spec:
    symbol: str
    tick: float
    spread: float            # typical spread, price units
    slip: float              # extra cost on stop fills, price units
    commission_per_million: float = 30.0   # USD per million USD traded, per side (0 for CFDs without commission)
    flat_commission_rt: float | None = None  # per unit round trip, overrides the per-million rate (cTrader backtest gold)
    swap_long: float = 0.0   # price units per unit per night (negative = you pay)
    swap_short: float = 0.0
    triple_day: int = 2      # weekday of the triple rollover: 2 = Wednesday, 4 = Friday
    approx_swaps: bool = False

    def commission_rt(self, price: float) -> float:
        if self.flat_commission_rt is not None:
            return self.flat_commission_rt
        return 2.0 * self.commission_per_million * 1e-6 * price


# Typical raw-account spreads (price units) and whether swaps below are approximations.
_TYPICAL = {
    "EURUSD": 0.00002, "GBPUSD": 0.00004, "USDJPY": 0.004, "AUDUSD": 0.00003, "USDCAD": 0.00004,
    "USDCHF": 0.00005, "NZDUSD": 0.00005, "EURGBP": 0.00005, "EURJPY": 0.006, "GBPJPY": 0.010,
    "AUDJPY": 0.006, "CADJPY": 0.008, "CHFJPY": 0.010, "NZDJPY": 0.008,
    "US500": 0.5, "USTEC": 1.0, "DE40": 1.2, "JP225": 6.0, "XAGUSD": 0.02, "XTIUSD": 0.03,
}

# Majors exported before symbols.csv was rewritten: approximate swaps in pips per night (long, short).
_APPROX_SWAP_PIPS = {
    "EURUSD": (-0.8, 0.3), "GBPUSD": (-0.5, -0.1), "USDJPY": (1.0, -2.0), "AUDUSD": (-0.4, -0.1),
    "USDCAD": (0.1, -0.6), "USDCHF": (0.8, -1.5), "NZDUSD": (-0.3, -0.2), "EURGBP": (-0.5, 0.0),
}


def _pip(symbol: str) -> float:
    if symbol.endswith("JPY"):
        return 0.01
    return 0.0001


def load_specs() -> dict[str, Spec]:
    specs: dict[str, Spec] = {}
    sheet = DATA_DIR / "symbols.csv"
    if sheet.exists():
        with open(sheet, newline="") as fh:
            for row in csv.DictReader(fh):
                sym = row["symbol"]
                pip = float(row["pip_size"])
                tick = float(row["tick_size"])
                spread = _TYPICAL.get(sym, max(float(row["spread_now"] or 0), tick))
                triple = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}.get(row["triple_swap_day"], 2)
                specs[sym] = Spec(
                    symbol=sym, tick=tick, spread=spread, slip=max(0.5 * spread, tick),
                    commission_per_million=float(row["commission"] or 0),
                    swap_long=float(row["swap_long"]) * pip, swap_short=float(row["swap_short"]) * pip,
                    triple_day=triple,
                )
    for sym, (sl, ss) in _APPROX_SWAP_PIPS.items():
        if sym not in specs:
            pip = _pip(sym)
            tick = pip / 10.0
            spread = _TYPICAL[sym]
            specs[sym] = Spec(sym, tick, spread, max(0.5 * spread, tick), 30.0, None, sl * pip, ss * pip, 2, True)
    # Gold, measured from the cTrader backtests (2026-09-30): flat $0.2508/oz round trip, fixed swaps,
    # about $0.20 slippage on stop fills; spread comes from the export itself.
    specs["XAUUSD"] = Spec("XAUUSD", 0.01, 0.15, 0.20, 30.0, 0.2508, -0.615, 0.431, 2)
    return specs
