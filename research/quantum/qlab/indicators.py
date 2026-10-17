"""The indicators QuantumTrendX and QuantumBreakoutX read, computed the way cTrader does:
EMA with alpha 2/(n+1), ATR as an exponential average of true range, and the Directional
Movement System with Wilder smoothing (alpha 1/n)."""
from __future__ import annotations

import numpy as np


def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    a = 1.0 / n
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev = np.concatenate(([c[0]], c[:-1]))
    return np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))


def atr_ema(h, l, c, n: int) -> np.ndarray:
    atr = ema(true_range(h, l, c), n)
    atr[:n] = np.nan   # cTrader shows no value before the period is filled
    return atr


def dms(h, l, c, n: int):
    """ADX, DI+ and DI- (Wilder)."""
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = wilder(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        dip = 100.0 * wilder(plus, n) / tr
        dim = 100.0 * wilder(minus, n) / tr
        dx = 100.0 * np.abs(dip - dim) / (dip + dim)
    dx = np.nan_to_num(dx)
    adx = wilder(dx, n)
    adx[: 2 * n] = np.nan
    return adx, dip, dim
