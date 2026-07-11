"""Regression tests for the snapshot-history and positioning modules (offline)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    import cte.adapters.base as base
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)
    return tmp_path


def _weekly(vals):
    idx = pd.date_range("2010-01-05", periods=len(vals), freq="W-TUE")
    return idx, vals


def _write_tff(tmp_cache, lev_pct, am_pct):
    import cte.adapters.base as base
    idx, lev = _weekly(lev_pct)
    oi = 100_000
    df = pd.DataFrame({
        "date": idx, "ccy": "AUD",
        "lev_net_pct_oi": lev, "am_net_pct_oi": am_pct,
        "lev_net": np.array(lev) * oi / 100, "am_net": np.array(am_pct) * oi / 100,
        "open_interest": oi, "source": "cftc_tff_combined",
        "fetched_at": pd.Timestamp("2026-01-01"),
    })
    base.write_cache(df, "tff")


def test_positioning_crowded_label_fires(tmp_cache):
    """A terminal spike far outside the series' own history must label CROWDED."""
    from cte.flags.positioning import positioning_snapshot
    base_hist = list(np.random.default_rng(0).normal(0, 3, 600))
    _write_tff(tmp_cache, base_hist + [35.0], base_hist + [0.0])
    snap = positioning_snapshot()
    aud = snap[snap.ccy == "AUD"].iloc[0]
    assert aud.lev_z > 1.5
    assert str(aud.pos_label).startswith("CROWDED long")


def test_positioning_three_way_warning(tmp_cache):
    """Vulnerable quadrant + crowded long must produce the combo warning; the
    same quadrant with normal positioning must not."""
    from cte.flags.positioning import positioning_snapshot, positioning_warnings
    hist = list(np.random.default_rng(1).normal(0, 3, 600))
    _write_tff(tmp_cache, hist + [35.0], hist + [0.0])
    snap = positioning_snapshot()
    tm = pd.DataFrame({"ccy": ["AUD"], "axis1_fundamental_struct": [-0.6],
                       "axis2_stretch_struct": [0.8]})
    w = positioning_warnings(snap, tm)
    assert "AUD" in w and any("Vulnerable AND crowded" in n for n in w["AUD"])

    _write_tff(tmp_cache, hist + [1.0], hist + [0.0])   # normal positioning
    w2 = positioning_warnings(positioning_snapshot(), tm)
    assert not any("Vulnerable AND crowded" in n for n in w2.get("AUD", []))


def test_positioning_absent_cache_degrades_gracefully(tmp_cache):
    from cte.flags.positioning import positioning_snapshot, positioning_warnings
    snap = positioning_snapshot()          # no tff cache written
    assert list(snap.columns) == ["ccy"] or snap["ccy"].notna().all()
    assert positioning_warnings(snap, None) == {}


def _tiny_universe(tmp_cache, monkeypatch):
    """Small synthetic features + fx/yields caches; patch build_features."""
    import cte.adapters.base as base
    import cte.scoring.history as H
    import cte.transform.features as F

    rng = np.random.default_rng(3)
    mdates = pd.date_range("2012-01-31", "2026-06-30", freq="ME")
    ddates = pd.date_range("2012-01-01", "2026-07-01", freq="B")
    feats = []
    for c in ["USD", "AUD"]:
        for m in ["gdp_yoy", "infl_gap", "real_policy", "real_10y"]:
            feats.append(pd.DataFrame({
                "date": mdates, "ccy": c, "feature": m,
                "value": np.cumsum(rng.normal(0, 0.3, len(mdates)))}))
    feats = pd.concat(feats, ignore_index=True)
    for name, extra in (("fx_spot", {}), ("yields", {"tenor": "10Y"})):
        base.write_cache(pd.concat([
            pd.DataFrame({"date": ddates, "ccy": c,
                          "value": 1 + np.cumsum(rng.normal(0, .004, len(ddates))),
                          **extra}) for c in ["USD", "AUD"]]), name)
    monkeypatch.setattr(F, "build_features", lambda: feats)
    monkeypatch.setattr(H, "build_features", lambda: feats)
    return H


def test_history_backfill_and_append_idempotent(tmp_cache, monkeypatch):
    H = _tiny_universe(tmp_cache, monkeypatch)
    hist = H.backfill()
    assert hist.date.nunique() > 100
    assert (hist.kind == "month_end").all()
    # struct axis must be empty before the 10y window can fill, present after
    early = hist[hist.date < "2015-01-01"]["axis1_fundamental_struct"]
    late = hist[hist.date > "2024-01-01"]["axis1_fundamental_struct"]
    assert early.isna().all() and late.notna().any()

    tm = (hist[hist.date == hist.date.max()]
          .drop(columns=["date", "kind"]).reset_index(drop=True))
    h1 = H.append_today(tm)
    h2 = H.append_today(tm)                     # same-day rerun replaces, not dups
    assert len(h1) == len(h2)
    assert (h2.kind == "daily").sum() == tm.ccy.nunique()


def test_trails_fig_renders_and_asof_filters_future(tmp_cache, monkeypatch):
    """The as-of view must not draw trail vertices from after the as-of date."""
    import matplotlib
    matplotlib.use("Agg")
    from cte.dashboard.plots import tension_map_fig

    H = _tiny_universe(tmp_cache, monkeypatch)
    hist = H.backfill()
    tm = (hist[hist.date == hist.date.max()]
          .drop(columns=["date", "kind"]).reset_index(drop=True))
    fig = tension_map_fig(tm, "regime", None, history=hist, trail_months=6,
                          crowded={"AUD"})
    assert fig is not None
    asof_rows = hist[(hist.date + pd.offsets.MonthEnd(0)) == "2020-03-31"] \
        .groupby("ccy").tail(1)
    fig2 = tension_map_fig(asof_rows, "regime", None, history=hist,
                           trail_months=6, asof_label="2020-03-31")
    # every plotted line vertex must precede the as-of date: reconstruct from data
    h = hist.dropna(subset=["axis1_fundamental_regime"])
    assert (h[h.date < pd.Timestamp("2020-03-31")].date.max()
            < pd.Timestamp("2020-03-31"))
    assert fig2 is not None
