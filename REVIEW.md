# Code Review Brief — Currency Tension Engine

You are reviewing this repo for **correctness and robustness, not redesign.** Please
read this whole file first. Several choices below are *deliberate* — do not "fix" them;
only tell me if they're implemented incorrectly.

## What it is

A daily macro dashboard that places 8 currencies (USD, EUR, JPY, GBP, CHF, CAD, AUD,
NZD) on a two-axis map — **Axis 1: fundamental trajectory** (deteriorating ↔ improving)
× **Axis 2: valuation & stretch** (cheap ↔ maxed-out) — plus a pairwise carry grid and
objective "inflection" overlays. Each currency is scored against **its own history** on
two horizons (~10y structural, ~2y regime).

## Data flow

```
adapters/  (ingest, per-source)          → tidy [date, ccy, metric, value]
  macro.build_macro_backbone()           unifies the fundamental panel
scripts/backfill.py + store.py           full-history seed + idempotent daily merge → parquet cache
transform/features.py                    raw panel → 14 normalized pillar features
transform/zscore.py                      dual-horizon calendar-window z-scores
transform/pairwise.py                    pairwise 2Y carry grid (real & nominal)
flags/overlays.py                        objective inflection gauges + warnings
scoring/compositor.py                    signed features → pillars → 2 axes (applies overlays)
scoring/engine.py                        assembles + persists the snapshot
commentary/narrator.py                   change-driven daily note (batch, cached)
dashboard/plots.py + streamlit_app.py    render from the cached snapshot
```

## File map (review priority)

**HIGH — logic and likely-bug surface (~880 lines total):**
- `cte/transform/features.py` — feature derivation, frequency-aware changes, cross-source date alignment
- `cte/transform/zscore.py` — dual-horizon rolling z-score
- `cte/transform/pairwise.py` — carry grid
- `cte/scoring/compositor.py` — pillar/axis composition + overlay application
- `cte/flags/overlays.py` — FX/yield correlation, hike-feasibility, carry-to-vol, warnings
- `cte/scoring/engine.py` — snapshot assembly/persist
- `cte/adapters/macro.py` — panel unification
- `cte/adapters/external.py` — current account + NIIP (annualization, unit matching)

**MEDIUM:** other `cte/adapters/*` (each is one source → tidy frame), `cte/store.py`,
`scripts/backfill.py`, `cte/config.py` (all the mapping/weight/sign tables).

**LOW:** `cte/dashboard/plots.py`, `streamlit_app.py`, `cte/commentary/narrator.py`.

## DELIBERATE — do not "fix," only flag if implemented wrong

1. **Carry is pairwise-only.** It lives in the grid (`pairwise.py`), NOT as a
   per-currency pillar. It was removed from the composite on purpose (a single-currency
   carry z-score double-counted and imported a common regime component).
2. **Each-currency-vs-own-history z-scores. No cross-sectional demeaning.** By design —
   the cross-sectional view is the carry grid. Don't propose demeaning the map.
3. **Pillar "Real 10Y (D)" is the market real yield, not the budget deficit.** The fiscal
   deficit is deliberately excluded (annual, stagnant, largely already in the yield).
4. **Inflation and Real-10Y signs are conditional, by design.** They're bent by objective
   overlays: inflation × `tanh(hike_feasibility)` (gated to above-target only), real_10y ×
   `tanh(FX-yield corr)`. This non-monotonic "supportive until an inflection" behavior is
   the whole point — not a bug.
5. **Empirical sourcing choices are verified and intentional:** headline CPI (not core),
   BCICP business-confidence (replaced composite CLI), real-GDP *levels* → YoY, BIS policy
   rates uniform, current account + NIIP both in External (C). See `docs/DATA_SOURCES.md`.
6. **Unemployment is multi-sourced for freshness:** FRED (USD), Eurostat (EUR), ONS (GBP),
   OECD (rest). ONS == OECD harmonised for the UK (verified 0.016pp mean diff), chosen for
   1–2mo fresher data. Not an inconsistency.
7. **Yield curves stay on each source's published convention** (mixed zero/par). NOT
   bootstrapped to uniform zero-coupon on purpose (bootstrapping adds more error than the
   ~1–2bp it removes at 2Y).
8. **Positioning (CFTC TFF) is a flag, not a pillar** — futures-only, lagged, thin.
9. **Commentary is batch-cached** (generated once/day by the Action; the public app never
   calls the API). The anthropic SDK is lazy-imported. Don't move generation into the app.
10. **Dates are snapped to period-end** across sources on purpose (FRED labels period-start,
    OECD period-end) so cross-source joins align.

## What I actually want from you

Focus on **bugs and silent-failure risks** — things that produce a *wrong number without
erroring*:
- Look-ahead / leakage in the rolling z-scores (`zscore.py`): confirm the window is strictly
  trailing and `min_periods` can't leak future data.
- Frequency/lag correctness in `features.py`: the `_chg`/`_pct` lag selection for mixed
  monthly/quarterly series; the month-end reindex + `ffill` of CPI onto the yield grid
  (any forward-fill leakage?); handling of sparse quarterly series in the wide pivot.
- Overlay application in `compositor._apply_overlays`: are multipliers applied to the exact
  right (ccy, metric) rows; is the `value > 0` inflation gate correct.
- `overlays.py`: FX-return vs yield-change correlation sign (note `fx_spot` value is USD per
  unit; the USD row is DXY); carry-to-vol percentile with short history; growth/policy z build.
- `external.py`: NIIP annualization (trailing-4Q nominal GDP), XDC unit matching, EA20 join.
- `pairwise.py`: grid antisymmetry (cell[i,j] == −cell[j,i]).
- Robustness: what each adapter and the compositor do on empty / stale / malformed input,
  and whether `store.merge_cache` is truly idempotent.
- Performance: anything that won't scale in the daily Action.

## How to report

- **Point to `file:line`** and give a **minimal repro or a failing test** for each claimed
  bug — don't assert bugs you can't demonstrate. You cannot run the pipeline end-to-end
  without API keys, so reason from the code and say when you're uncertain.
- **Separate "bug" from "design question."** If something looks wrong but might be one of
  the deliberate choices above, ask rather than assume.
- Rank findings by severity (wrong-number > crash > robustness > style).

Reference docs in the repo: `currency_engine_spec_v4.md` (the spec / source of truth) and
`docs/DATA_SOURCES.md` (every input → pillar → source → freshness). These encode the
empirical decisions; don't "correct" verified sourcing without flagging it as a question.
