# Quantum research lab (inside Vibe-Trading)

Research for the cTrader bots in `Ctrader Scalper` (QuantumTrendX, QuantumBreakoutX, QuantumHedge),
kept in this fork so it is versioned and cannot be lost again.

## What is here

| File | What it does |
|---|---|
| `qlab/data.py` | Loads the broker's M5 history (cTrader exports by the QuantumDataExport cBot in `Documents\QuantumData`, plus the XAUUSD export log) and builds H1/H4 bars on UTC boundaries. |
| `qlab/costs.py` | Spread, commission, swaps and stop slippage per market, from the broker's own symbol sheet. |
| `qlab/indicators.py` | EMA, ATR (exponential) and ADX/DI (Wilder), as cTrader computes them. |
| `qlab/models.py` | QuantumTrendX and QuantumBreakoutX, rule for rule from the C# bots, walking the M5 price path inside every hour. |
| `qlab/validate.py` | Feeds our trades into Vibe-Trading's own validation (`agent/backtest/validation.py`): bootstrap Sharpe confidence interval, walk-forward windows, Monte Carlo reshuffle. |
| `run_calibration.py` | Step 1 - checks the simulator against the cTrader backtests on gold. |
| `run_markets.py` | Step 2 - both models, default settings, on every market, with a pre-registered pass rule. |
| `run_portfolio.py` | Step 3 - combines the streams that passed and compares them with gold QTX alone. |

Run from this folder with Python 3.11+ (numpy, pandas): `python run_calibration.py`, `python run_markets.py`,
`python run_portfolio.py`. No API key or LLM is needed; the cached data goes to `.cache/` (not in git).

## Method

- **Data:** the user's own broker (IC Markets / Raw Trading cTrader), M5 bid bars from 2020-01-01.
- **Costs:** commission $30/M per side for FX and metals (gold: the flat $0.2508/oz round trip cTrader's
  backtester charges), today's swap rates applied to the whole history (as cTrader backtests do),
  typical raw spreads where the export has none, and extra slippage on every stop fill.
- **Calibration (step 1):** the simulator must reproduce the cTrader runs we already have before any of its
  numbers are trusted.
- **No tuning:** every market runs the same default settings. A market passes only if it is profitable
  in both periods under a rule fixed before the results were seen.
- **Two periods:** 2020-2023 (discovery) and 2024-Sep 2026 (held out).
- **Final check:** anything recommended still needs a cTrader backtest on the user's account before
  real money - the simulator runs a little optimistic (about +10R on gold QTX over 6.7 years).

## Results (2026-10-18)

**Step 1 - calibration on gold (2020 - Sep 2026):** QTX 599 trades, +69R, PF 1.28, DD 11.9R vs cTrader
590-597 trades, +57-60R, PF 1.22-1.25. QBX 1,164 trades, +65R, PF 1.12 vs cTrader 1,166, +64-72R,
PF 1.12-1.14, same losing years. Close enough to compare ideas; about +10R optimistic on gold QTX.

**Step 2 - 21 markets x 2 models (`results_markets.csv`):** 5 of 42 passed the pre-registered rule:
XAUUSD QTX, XAGUSD QTX, USDJPY QTX, USDJPY QBX, USTEC QBX. Most FX pairs, US500, DE40, JP225 and oil
failed, as the earlier FX study found.

**Step 2c - selection luck (`run_selection_check.py`):** choosing on 2020-23 alone picks 10 streams,
which together broke even in 2024-26 (PF 1.01, +23R, DD 89R). Choosing markets by past results is not
a strategy by itself; every candidate needs more evidence.

**Step 2b - robustness (`results_robustness.csv`):** random entries with the same exits beat the real
signal in 1/40 runs for XAUUSD QTX, XAGUSD QTX and USDJPY QBX, 2/40 for USTEC QBX and 3/40 for USDJPY QTX
(dropped: not convincing, and 0.49 correlated with USDJPY QBX). All five stay profitable with doubled
spreads/slippage and no swap credits, and 8/8 neighbouring settings are profitable in both periods.

**Step 3 - portfolio (`results_portfolio_final4.csv`):** XAUUSD QTX + XAGUSD QTX + USDJPY QBX + USTEC QBX,
monthly correlations -0.18 to +0.42:

| | Gold QTX alone | Portfolio 4 |
|---|---|---|
| Net | +69R | +351R |
| Max DD | 11.9R | 25.8R |
| Return / DD | 5.8 | 13.6 |
| Losing years | 2024 | none |
| Sharpe (Vibe bootstrap, 95% CI) | 0.54 (-0.22 to 1.18) | 1.43 (0.76 to 2.09) |
| Walk-forward windows profitable | 5/6 | 6/6 |

At 0.5% risk per trade the simulated max DD is about 12%. QuantumHedge's correlation filter at 0.80
would block 176 silver/gold trades (-21R); 0.90 keeps nearly all of them.

**Not proven:** the four markets were chosen on results (see step 2c), USDJPY swaps are approximate,
and the simulator runs a little optimistic. Next: cTrader backtest of the preset, then a demo period.
