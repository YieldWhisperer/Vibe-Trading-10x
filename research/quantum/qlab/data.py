"""Broker price data for the Quantum research: the cTrader M5 exports in Documents\\QuantumData
(one CSV per symbol, written by the QuantumDataExport cBot) and the older XAUUSD export log.

Every series is bid prices in UTC. H1 and H4 bars are built from the M5 bars on UTC
boundaries (as cTrader builds them), and each H1 bar keeps the range of M5 rows inside it so
the simulators can walk the price path within the hour.
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
DATA_DIR = Path(os.environ.get("QUANTUM_DATA", HOME / "Documents" / "QuantumData"))
GOLD_LOG = Path(os.environ.get(
    "QUANTUM_GOLD_LOG",
    HOME / "Documents" / "cAlgo" / "Data" / "cBots" / "QuantumDataExport"
    / "17be5a89-5f50-40dd-8b44-43144806d364-Default" / "Backtesting" / "log.txt",
))
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


@dataclass
class Market:
    """M5 rows plus the H1 and H4 bars built from them."""

    symbol: str
    m5: pd.DataFrame          # index: UTC open time; open high low close spread (spread NaN when unknown)
    h1: pd.DataFrame          # open high low close + first/last: M5 row range [first, last)
    h4: pd.DataFrame          # open high low close


def available_symbols() -> list[str]:
    names = sorted(p.name.split("_Minute5")[0] for p in DATA_DIR.glob("*_Minute5.csv"))
    if GOLD_LOG.exists() and "XAUUSD" not in names:
        names.append("XAUUSD")
    return names


def _read_gold_log(path: Path) -> pd.DataFrame:
    rows = []
    pat = re.compile(r"\| BAR,(\d{4}-\d\d-\d\d \d\d:\d\d),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,\s]*)")
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                rows.append(m.groups())
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "spread"])
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M")
    for c in ["open", "high", "low", "close", "volume", "spread"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("time")


def load_m5(symbol: str) -> pd.DataFrame:
    """M5 bid bars for a symbol (cached as a pickle after the first read)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}_m5.pkl"
    csv = DATA_DIR / f"{symbol}_Minute5.csv"
    source = csv if csv.exists() else (GOLD_LOG if symbol == "XAUUSD" else None)
    if source is None or not source.exists():
        raise FileNotFoundError(f"no M5 data for {symbol} in {DATA_DIR}")
    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    if source == csv:
        df = pd.read_csv(csv)
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M")
        df = df.set_index("time")
        df["spread"] = pd.to_numeric(df.get("spread"), errors="coerce")
    else:
        df = _read_gold_log(source)

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[["open", "high", "low", "close", "spread"]].astype(float)
    with open(cache, "wb") as fh:
        pickle.dump(df, fh)
    return df


def _bars(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = m5.resample(rule, label="left", closed="left")
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    return bars.dropna(subset=["open"])


def load_market(symbol: str, start: str | None = None, end: str | None = None) -> Market:
    m5 = load_m5(symbol)
    if start:
        m5 = m5[m5.index >= pd.Timestamp(start)]
    if end:
        m5 = m5[m5.index < pd.Timestamp(end)]

    h1 = _bars(m5, "1h")
    hour_of_row = m5.index.floor("h").values
    first = np.searchsorted(hour_of_row, h1.index.values, side="left")
    last = np.searchsorted(hour_of_row, h1.index.values, side="right")
    h1["first"] = first
    h1["last"] = last
    h4 = _bars(m5, "4h")
    return Market(symbol, m5, h1, h4)
