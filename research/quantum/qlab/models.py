"""QuantumTrendX (QTX) and QuantumBreakoutX (QBX) simulators.

Rule for rule from the cBots (QuantumHedge/Core: TrendXRules, MarketState, SignalBook,
TradeManager, SessionClock). Signals are read on each H1 close; inside each hour the price walks
through the M5 bars (open -> the adverse extreme -> the favourable extreme -> close), so stops,
targets, break-even, profit locks and trails trigger in a realistic order.

Every trade's result is in R: 1R = the initial stop distance. Costs (spread, commission, swaps,
slippage on stop fills) are taken out of the net R.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import indicators as ind
from .costs import Spec
from .data import Market


@dataclass
class QtxParams:
    channel: int = 20
    buffer_atr: float = 0.10
    fast: int = 20
    slow: int = 100
    adx_period: int = 14
    adx_min: float = 15.0
    use_bias: bool = True
    atr_period: int = 14
    baseline: int = 100
    dead_market: float = 0.5
    big_move: bool = True
    big_move_bars: int = 10
    big_move_atr: float = 1.5
    max_spread_atr: float = 0.15
    cooldown: int = 2
    stop_atr: float = 2.5
    target_r: float = 3.0
    profit_lock: bool = True
    locks: tuple = ((1.0, 0.3), (2.0, 1.0), (3.0, 2.0))
    break_even: bool = True
    break_even_r: float = 0.75
    runner: bool = True
    runner_atr: float = 2.5
    tighten_r: float = 6.0
    tighten_factor: float = 0.6
    loss_streak: int = 5
    loss_pause_min: int = 1440
    direction: int = 0          # 0 both, 1 long only, -1 short only


@dataclass
class QbxParams:
    lookback: int = 20
    stop_atr: float = 2.0
    target_r: float = 4.0
    break_even: bool = True
    break_even_r: float = 1.0
    trail: bool = True
    trail_bars: int = 3
    max_spread_atr: float = 0.15
    friday_rules: bool = True
    friday_no_entry_ny: int = 12
    friday_close_ny: int = 16
    atr_period: int = 14


@dataclass
class Trade:
    symbol: str
    model: str
    direction: int
    entry_time: pd.Timestamp
    entry: float
    risk: float
    exit_time: pd.Timestamp | None = None
    exit: float = math.nan
    exit_kind: str = ""
    r_gross: float = 0.0
    r_net: float = 0.0
    costs_r: float = 0.0
    # management state
    stop: float = 0.0
    target: float | None = None
    lock: int = 0          # 0 none, 1 break-even, 2..4 lock levels 1..3
    trail: bool = False


# ── New York clock, swaps ────────────────────────────────────────────────────

def _nth_sunday(year: int, month: int, n: int) -> datetime:
    first = datetime(year, month, 1)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


_DST: dict[int, tuple[datetime, datetime]] = {}


def ny(utc: datetime) -> datetime:
    y = utc.year
    if y not in _DST:
        _DST[y] = (_nth_sunday(y, 3, 2) + timedelta(hours=7), _nth_sunday(y, 11, 1) + timedelta(hours=6))
    s, e = _DST[y]
    return utc - timedelta(hours=4 if s <= utc < e else 5)


def weekend_window(utc: datetime, p: QbxParams) -> bool:
    t = ny(utc)
    if t.weekday() >= 5:
        return True
    return p.friday_rules and t.weekday() == 4 and t.hour >= p.friday_no_entry_ny


def friday_close(utc: datetime, p: QbxParams) -> bool:
    if not p.friday_rules:
        return False
    t = ny(utc)
    return t.weekday() == 4 and t.hour >= p.friday_close_ny


def rollovers(entry: datetime, exit: datetime, triple_day: int) -> int:
    """Rollover units between two moments: one per weekday at 17:00 New York, three on the triple day."""
    units = 0
    day = ny(entry).date()
    last = ny(exit).date()
    while day <= last:
        if day.weekday() < 5:
            roll_ny = datetime(day.year, day.month, day.day, 17)
            # back to UTC (NY offset at that moment)
            roll_utc = roll_ny + timedelta(hours=5)
            if ny(roll_utc).hour != 17:
                roll_utc = roll_ny + timedelta(hours=4)
            if entry < roll_utc <= exit:
                units += 3 if day.weekday() == triple_day else 1
        day += timedelta(days=1)
    return units


def _finish(t: Trade, spec: Spec, when, px: float, kind: str, slip: bool) -> Trade:
    t.exit_time = when
    t.exit = px
    t.exit_kind = kind
    t.r_gross = t.direction * (px - t.entry) / t.risk
    cost = spec.commission_rt(t.entry)
    if slip:
        cost += spec.slip
    nights = rollovers(t.entry_time.to_pydatetime(), when.to_pydatetime(), spec.triple_day)
    swap = (spec.swap_long if t.direction > 0 else spec.swap_short) * nights
    t.costs_r = (cost - swap) / t.risk
    t.r_net = t.r_gross - t.costs_r
    return t


# ── Shared bar readings ─────────────────────────────────────────────────────

class Readings:
    """Indicator readings on every closed H1 bar (index = the closed bar)."""

    def __init__(self, mk: Market, atr_period: int, q: QtxParams | None = None, qbx: QbxParams | None = None):
        h = mk.h1
        self.t = h.index
        self.o, self.hi, self.lo, self.c = (h[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.first = h["first"].to_numpy(int)
        self.last = h["last"].to_numpy(int)
        self.atr = ind.atr_ema(self.hi, self.lo, self.c, atr_period)
        n = len(self.c)
        if q is not None:
            self.fast = ind.ema(self.c, q.fast)
            self.slow = ind.ema(self.c, q.slow)
            self.adx, self.dip, self.dim = ind.dms(self.hi, self.lo, self.c, q.adx_period)
            a = pd.Series(np.where(self.atr > 0, self.atr, np.nan))
            self.baseline = a.rolling(q.baseline, min_periods=1).mean().to_numpy()
            self.chan_up = pd.Series(self.hi).rolling(q.channel).max().shift(1).to_numpy()
            self.chan_dn = pd.Series(self.lo).rolling(q.channel).min().shift(1).to_numpy()
            self.big = (self.c - np.concatenate((np.full(q.big_move_bars, np.nan), self.c[:-q.big_move_bars]))) / self.atr
            # higher-timeframe bias: H4 EMA(slow) slope over 3 closed H4 bars
            h4 = mk.h4
            e4 = ind.ema(h4["close"].to_numpy(float), q.slow)
            t4 = h4.index.values
            nxt = np.concatenate((self.t.values[1:], [self.t.values[-1] + np.timedelta64(1, "h")]))
            forming = np.searchsorted(t4, nxt, side="right") - 1
            closed = forming - 1
            bias = np.zeros(n, dtype=int)
            ok = closed - 3 >= 0
            ce = np.where(ok, e4[np.clip(closed, 0, None)], np.nan)
            pe = np.where(ok, e4[np.clip(closed - 3, 0, None)], np.nan)
            bias[(self.c > ce) & (ce >= pe)] = 1
            bias[(self.c < ce) & (ce <= pe)] = -1
            self.bias = bias
        if qbx is not None:
            self.level = pd.Series(self.hi).rolling(qbx.lookback).max().to_numpy()
            self.trail_low = pd.Series(self.lo).rolling(qbx.trail_bars).min().to_numpy()


def _spread_at(mk_spread: np.ndarray, j: int, default: float) -> float:
    s = mk_spread[j]
    return default if not (s == s) or s <= 0 else s


def _path(o, h, l, c, buy: bool):
    return (o, l, h, c) if buy else (o, h, l, c)


# ── QuantumTrendX ───────────────────────────────────────────────────────────

def _manage_qtx(t: Trade, bid: float, ask: float, spread: float, tick: float, atr: float, p: QtxParams, gap_point: bool):
    """One price. Returns (exit_price, kind) when the stop or target is reached, else None.

    Only an M5 open can gap through the stop (fill at that price); inside a bar the price passes
    the stop on its way to the extreme, so the fill is the stop itself."""
    buy = t.direction > 0
    if (bid <= t.stop) if buy else (ask >= t.stop):
        kind = "trail" if t.trail else ("lock" if t.lock >= 2 else ("break_even" if t.lock == 1 else "stop"))
        if not gap_point:
            return t.stop, kind
        return (min(bid, t.stop) if buy else max(ask, t.stop)), kind
    if t.target is not None and ((bid >= t.target) if buy else (ask <= t.target)):
        return t.target, "target"
    price = bid if buy else ask
    progress = t.direction * (price - t.entry) / t.risk
    gap = max(spread * 2.0, tick * 2.0)
    if p.profit_lock:
        level, lock_r = 0, 0.0
        for k, (trig, lk) in enumerate(p.locks):
            if progress >= trig:
                level, lock_r = k + 2, lk
        if level > t.lock:
            stop = t.entry + t.direction * lock_r * t.risk
            if (stop <= t.stop) if buy else (stop >= t.stop):
                t.lock = level
            elif not ((stop >= price - gap) if buy else (stop <= price + gap)):
                t.stop, t.lock, t.target = stop, level, None
                return None
    if p.break_even and t.lock < 1 and progress >= p.break_even_r:
        stop = t.entry + t.direction * spread
        if (t.stop >= stop) if buy else (t.stop <= stop):
            t.lock = 1
        elif not ((stop >= price - gap) if buy else (stop <= price + gap)):
            t.stop, t.lock = stop, 1
    if p.runner and t.lock >= 4 and atr > 0:
        dist = p.runner_atr * atr * (p.tighten_factor if progress >= p.tighten_r else 1.0)
        dist = max(dist, gap)
        stop = price - t.direction * dist
        step = max(0.1 * atr, tick)
        if (stop >= t.stop + step) if buy else (stop <= t.stop - step):
            t.stop, t.trail = stop, True
    return None


def sim_qtx(mk: Market, spec: Spec, p: QtxParams = QtxParams(), entries: dict[int, int] | None = None) -> list[Trade]:
    """entries: {H1 bar index: direction} replaces the signal (random-entry benchmark); management is unchanged."""
    r = Readings(mk, p.atr_period, q=p)
    m5 = mk.m5
    mo, mh, ml, mc = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ms = m5["spread"].to_numpy(float)
    mt = m5.index
    warm = max(max(p.channel + p.big_move_bars, p.slow), max(20, p.baseline)) + 5
    trades: list[Trade] = []
    t: Trade | None = None
    last_activity = -10**9
    streak = 0
    pause_until = None

    for i in range(warm + 1, len(r.c)):
        ci = i - 1
        j0 = r.first[i]
        bar_time = r.t[i]
        spread0 = _spread_at(ms, j0, spec.spread)
        bid0, ask0 = mo[j0], mo[j0] + spread0
        atr = r.atr[ci]

        # QuantumTrendX verdict on the bar that just closed (TrendXRules.Evaluate + SignalBook gates)
        side = 0
        if entries is not None:
            if t is None and atr > 0 and i in entries:
                side = entries[i]
                entry = ask0 if side > 0 else bid0
                risk = p.stop_atr * atr
                t = Trade(mk.symbol, "QTX", side, bar_time, entry, risk, stop=entry - side * risk, target=entry + side * p.target_r * risk)
            side = 0
        elif atr > 0 and r.baseline[ci] > 0:
            ratio = atr / r.baseline[ci]
            buf = p.buffer_atr * atr
            if r.c[ci] > r.chan_up[ci] + buf:
                side = 1
            elif r.c[ci] < r.chan_dn[ci] - buf:
                side = -1
            if side:
                trend = 0
                if r.adx[ci] == r.adx[ci] and r.adx[ci] >= p.adx_min:
                    if r.fast[ci] > r.slow[ci] and r.dip[ci] > r.dim[ci]:
                        trend = 1
                    elif r.fast[ci] < r.slow[ci] and r.dim[ci] > r.dip[ci]:
                        trend = -1
                if p.dead_market > 0 and ratio < p.dead_market:
                    side = 0
                elif trend != side:
                    side = 0
                elif p.use_bias and r.bias[ci] != 0 and r.bias[ci] != side:
                    side = 0
                elif p.big_move and side * r.big[ci] < p.big_move_atr:
                    side = 0
        if side and t is None:
            if pause_until is not None and bar_time < pause_until:
                side = 0
            elif p.max_spread_atr > 0 and spread0 > p.max_spread_atr * atr:
                side = 0
            elif p.cooldown > 0 and i - last_activity < p.cooldown:
                side = 0
            elif p.direction and side != p.direction:
                side = 0
            if side:
                entry = ask0 if side > 0 else bid0
                risk = p.stop_atr * atr
                t = Trade(mk.symbol, "QTX", side, bar_time, entry, risk, stop=entry - side * risk, target=entry + side * p.target_r * risk)
                last_activity = i

        if t is None:
            continue
        for j in range(j0, r.last[i]):
            sp = _spread_at(ms, j, spec.spread)
            done = None
            for k, px in enumerate(_path(mo[j], mh[j], ml[j], mc[j], t.direction > 0)):
                done = _manage_qtx(t, px, px + sp, sp, spec.tick, atr, p, k == 0)
                if done:
                    break
            if done:
                px, kind = done
                trades.append(_finish(t, spec, mt[j], px, kind, slip=kind != "target"))
                last_activity = i
                loss = t.r_gross < 0
                streak = streak + 1 if loss else 0
                if p.loss_streak > 0 and streak >= p.loss_streak:
                    streak = 0
                    pause_until = bar_time + pd.Timedelta(minutes=p.loss_pause_min) if p.loss_pause_min > 0 else bar_time.normalize() + pd.Timedelta(days=1)
                t = None
                break
    return trades


# ── QuantumBreakoutX ────────────────────────────────────────────────────────

def sim_qbx(mk: Market, spec: Spec, p: QbxParams = QbxParams(), entries: dict[int, int] | None = None) -> list[Trade]:
    """entries: {H1 bar index: 1} buys at those bar opens instead of the breakout level (random-entry benchmark)."""
    r = Readings(mk, p.atr_period, qbx=p)
    m5 = mk.m5
    mo, mh, ml, mc = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ms = m5["spread"].to_numpy(float)
    mt = m5.index
    mt_py = mt.to_pydatetime()
    warm = 105
    trades: list[Trade] = []
    t: Trade | None = None

    for i in range(warm + 1, len(r.c)):
        ci = i - 1
        j0 = r.first[i]
        bar_time = r.t[i].to_pydatetime()
        spread0 = _spread_at(ms, j0, spec.spread)
        bid0, ask0 = mo[j0], mo[j0] + spread0
        atr = r.atr[ci]
        gap0 = max(2 * spread0, 2 * spec.tick)

        if t is not None:
            if friday_close(bar_time, p):
                trades.append(_finish(t, spec, r.t[i], bid0, "friday", slip=False))
                t = None
            elif p.trail and t.lock >= 1:
                stop = r.trail_low[ci]
                if stop > t.stop + spec.tick and stop < bid0 - gap0:
                    t.stop, t.trail = stop, True

        level = math.nan
        risk = 0.0
        if entries is not None:
            if t is None and i in entries and not weekend_window(bar_time, p) and atr > 0:
                risk = p.stop_atr * atr
                t = Trade(mk.symbol, "QBX", 1, r.t[i], ask0, risk, stop=ask0 - risk, target=ask0 + p.target_r * risk if p.target_r > 0 else None)
        elif t is None and not weekend_window(bar_time, p) and atr > 0 and not (p.max_spread_atr > 0 and spread0 > p.max_spread_atr * atr):
            level = r.level[ci]
            risk = p.stop_atr * atr
            if ask0 >= level:
                t = Trade(mk.symbol, "QBX", 1, r.t[i], ask0, risk, stop=ask0 - risk, target=ask0 + p.target_r * risk if p.target_r > 0 else None)
                level = math.nan

        for j in range(j0, r.last[i]):
            sp = _spread_at(ms, j, spec.spread)
            for k, px in enumerate((mo[j], ml[j], mh[j], mc[j])):
                bid, ask = px, px + sp
                if t is not None:
                    if friday_close(mt_py[j], p):
                        trades.append(_finish(t, spec, mt[j], bid, "friday", slip=False))
                        t = None
                    else:
                        done = None
                        if bid <= t.stop:
                            done = (min(bid, t.stop) if k == 0 else t.stop, "trail" if t.trail else ("break_even" if t.lock else "stop"))
                        elif t.target is not None and bid >= t.target:
                            done = (t.target, "target")
                        elif p.break_even and t.lock < 1 and (bid - t.entry) / t.risk >= p.break_even_r:
                            stop = t.entry + 2 * spec.tick
                            gap = max(2 * sp, 2 * spec.tick)
                            if t.stop >= stop:
                                t.lock = 1
                            elif stop < bid - gap:
                                t.stop, t.lock = stop, 1
                        if done:
                            trades.append(_finish(t, spec, mt[j], done[0], done[1], slip=done[1] != "target"))
                            t = None
                elif level == level:
                    if weekend_window(mt_py[j], p):
                        level = math.nan
                    elif ask >= level:
                        fill = (ask if k == 0 else level) + spec.slip
                        t = Trade(mk.symbol, "QBX", 1, mt[j], fill, risk, stop=fill - risk, target=fill + p.target_r * risk if p.target_r > 0 else None)
                        level = math.nan
    return trades


# ── Summaries ──────────────────────────────────────────────────────────────

def to_frame(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": t.symbol, "model": t.model, "direction": t.direction, "entry_time": t.entry_time,
        "exit_time": t.exit_time, "entry": t.entry, "exit": t.exit, "risk": t.risk, "exit_kind": t.exit_kind,
        "r_gross": t.r_gross, "r_net": t.r_net, "costs_r": t.costs_r,
    } for t in trades])


def summary(df: pd.DataFrame, col: str = "r_net") -> dict:
    if df.empty:
        return {"trades": 0, "total_r": 0.0, "pf": float("nan"), "max_dd_r": 0.0, "win_rate": float("nan")}
    r = df[col].to_numpy()
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.concatenate(([0.0], eq))) [1:] - eq)) if len(eq) else 0.0
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": int(len(r)), "total_r": float(r.sum()), "pf": float(pos / neg) if neg > 0 else float("inf"),
            "max_dd_r": dd, "win_rate": float((r > 0).mean())}
