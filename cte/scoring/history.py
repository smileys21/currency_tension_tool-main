"""Snapshot history — the tension map through time (trails, time dial, backtests).

Two entry points:

  backfill()      one-time: recompute the map as-of every month-end back through the
                  data (~2007+), using AS-OF overlay multipliers, not today's — the
                  yield-reward correlation and hike-feasibility are themselves rolling
                  series, so each historical point is bent by what the overlays said
                  THEN. Writes/overwrites data/cache/snapshot_history.parquet.

  append_today()  daily: called by the engine after build_snapshot; appends today's
                  live axis scores (idempotent per calendar date).

Honesty note (docs/DATA_SOURCES.md carries the long form): the backfill is computed
from TODAY'S vintage of the data — revised macro, and CPI aligned to its reference
month. Historical positions are "as we'd compute them now", not "as you'd have seen
them then". Fine for trails and the dial; a point-in-time backtest must additionally
lag inflation-derived features by one month and treat revised macro with suspicion.

Grain: month-end rows from the backfill (kind='month_end'), daily rows from the
engine (kind='daily'). Trails resample to month-end; the dial exposes every row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.adapters.base import read_cache, utcnow, write_cache
from cte.config import CURRENCIES
from cte.flags.overlays import _fx_wide, _y10_wide, _WIN
from cte.scoring.compositor import score
from cte.transform.features import build_features
from cte.transform.zscore import dual_horizon_z

HISTORY_NAME = "snapshot_history"
_STALE_LIMIT = 12          # month-ends a feature may be carried forward as-of
_AXIS_COLS = ["axis1_fundamental_struct", "axis2_stretch_struct",
              "axis1_fundamental_regime", "axis2_stretch_regime"]


# ------------------------------------------------------------------ as-of panels

def _monthly_panel(z: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict]:
    """Reindex each (ccy, metric) z-series onto a common month-end grid with a
    bounded forward-fill, so 'as-of date D' is one slice, not N searches."""
    grid = pd.date_range(z.date.min() + pd.offsets.MonthEnd(0),
                         z.date.max() + pd.offsets.MonthEnd(0), freq="ME")
    panels = {}
    for (ccy, metric), g in z.groupby(["ccy", "metric"]):
        g = g.sort_values("date").set_index("date")
        g = g[~g.index.duplicated(keep="last")]
        g.index = g.index + pd.offsets.MonthEnd(0)
        g = g[~g.index.duplicated(keep="last")]
        panels[(ccy, metric)] = (g[["value", "struct_z", "regime_z"]]
                                 .reindex(grid).ffill(limit=_STALE_LIMIT))
    return grid, panels


def _overlay_history(z_grid: pd.DatetimeIndex,
                     panels: dict) -> dict[pd.Timestamp, pd.DataFrame]:
    """As-of overlay multipliers per month-end: real10y_mult from the rolling
    FX/yield correlation series (month-end sampled), infl_mult from as-of
    growth/policy z (same construction as overlays.hike_feasibility, applied
    through time)."""
    # real10y_mult: full corr series per ccy, month-end sampled
    fx, y10 = _fx_wide(), _y10_wide()
    r10 = {}
    for c in CURRENCIES:
        if c not in fx or c not in y10:
            continue
        j = pd.concat({"fx": fx[c], "y": y10[c]}, axis=1, join="inner").dropna()
        corr = j["fx"].pct_change().rolling(_WIN).corr(j["y"].diff())
        corr.index = corr.index + pd.offsets.MonthEnd(0)
        corr = corr[~corr.index.duplicated(keep="last")]
        r10[c] = np.tanh(2.0 * corr.reindex(z_grid).ffill(limit=2))

    # infl_mult: growth composite minus real-policy restrictiveness, as-of
    g_feats = {"bcicp_slope": 1, "gdp_yoy": 1, "unemp_3m_chg": -1}
    out = {}
    for d in z_grid:
        rows = []
        for c in CURRENCIES:
            gz = [panels[(c, m)].loc[d, "struct_z"] * s
                  for m, s in g_feats.items() if (c, m) in panels]
            gz = [v for v in gz if pd.notna(v)]
            growth = float(np.mean(gz)) if gz else np.nan
            rp = (panels[(c, "real_policy")].loc[d, "struct_z"]
                  if (c, "real_policy") in panels else np.nan)
            feas = (0 if pd.isna(growth) else growth) - \
                   (0 if pd.isna(rp) else rp)
            imult = np.tanh(feas) if (pd.notna(growth) or pd.notna(rp)) else np.nan
            rmult = float(r10[c].loc[d]) if c in r10 and pd.notna(r10[c].loc[d]) \
                    else np.nan
            rows.append({"ccy": c, "real10y_mult": rmult, "infl_mult": imult})
        out[d] = pd.DataFrame(rows)
    return out


def _asof_frame(panels: dict, d: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for (ccy, metric), p in panels.items():
        r = p.loc[d]
        if pd.isna(r["value"]) and pd.isna(r["struct_z"]) and pd.isna(r["regime_z"]):
            continue
        rows.append({"ccy": ccy, "metric": metric, "value": r["value"],
                     "struct_z": r["struct_z"], "regime_z": r["regime_z"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ entry points

def backfill(persist: bool = True) -> pd.DataFrame:
    """Recompute the tension map as-of every month-end. Minutes, not hours; run
    once (the Action seeds it when the committed file is absent)."""
    feats = build_features().rename(columns={"feature": "metric"})
    z = dual_horizon_z(feats)
    grid, panels = _monthly_panel(z)
    overlays = _overlay_history(grid, panels)

    out = []
    for d in grid:
        asof = _asof_frame(panels, d)
        if asof.empty:
            continue
        snap = overlays[d]
        row = {}
        for horizon, zcol in (("struct", "struct_z"), ("regime", "regime_z")):
            # early dates have no z on this horizon yet (window not filled) —
            # score() can't composite an empty frame, so skip the horizon
            if not asof[zcol].notna().any():
                continue
            _, axis = score(zcol, asof, snap)
            for _, r in axis.iterrows():
                row.setdefault(r["ccy"], {})[f"{r['axis']}_{horizon}"] = r["ascore"]
        for ccy, vals in row.items():
            if any(pd.notna(v) for v in vals.values()):
                out.append({"date": d, "ccy": ccy, "kind": "month_end", **vals})

    hist = pd.DataFrame(out)
    for c in _AXIS_COLS:
        if c not in hist.columns:
            hist[c] = np.nan
    hist = hist[["date", "ccy", "kind"] + _AXIS_COLS].sort_values(["date", "ccy"])
    if persist:
        write_cache(hist.reset_index(drop=True), HISTORY_NAME)
    return hist


def append_today(tm: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Append the live snapshot's axis scores under today's date. Idempotent:
    re-running on the same date replaces that date's daily rows."""
    asof = (asof or utcnow()).normalize()
    hist = read_cache(HISTORY_NAME)
    add = tm.copy()
    add["date"], add["kind"] = asof, "daily"
    for c in _AXIS_COLS:
        if c not in add.columns:
            add[c] = np.nan
    add = add[["date", "ccy", "kind"] + _AXIS_COLS]
    if hist is None:
        out = add
    else:
        keep = hist[~((hist.date == asof) & (hist.kind == "daily"))]
        out = pd.concat([keep, add], ignore_index=True)
    out = out.sort_values(["date", "ccy"]).reset_index(drop=True)
    write_cache(out, HISTORY_NAME)
    return out


def load_history() -> pd.DataFrame | None:
    return read_cache(HISTORY_NAME)


def dial_options(hist: pd.DataFrame, horizon: str,
                 min_ccys: int = 4) -> list[pd.Timestamp]:
    """Month-ends the time dial may offer for a horizon: completed month-end rows
    only (kind == 'month_end'; daily appends are the LIVE terminal point, never a
    historical option — otherwise the current partial month appears as a phantom
    future-dated month-end), where at least min_ccys currencies have both axis
    scores (struct z's start ~8y after regime z's)."""
    if hist is None or not len(hist):
        return []
    h = hist[hist.kind == "month_end"] if "kind" in hist.columns else hist
    cols = [f"axis1_fundamental_{horizon}", f"axis2_stretch_{horizon}"]
    v = h.dropna(subset=[c for c in cols if c in h.columns])
    if v.empty:
        return []
    cnt = v.groupby(v.date + pd.offsets.MonthEnd(0))["ccy"].nunique()
    return sorted(cnt[cnt >= min_ccys].index)


if __name__ == "__main__":
    h = backfill()
    print(f"backfilled {h.date.nunique()} month-ends x "
          f"{h.ccy.nunique()} currencies -> {len(h)} rows "
          f"({h.date.min().date()} -> {h.date.max().date()})")
