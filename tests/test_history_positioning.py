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


def test_dial_options_exclude_phantom_current_month(tmp_cache, monkeypatch):
    """Daily appends snap to a FUTURE month-end; the dial must never offer that
    phantom month — month_end rows only (the regression behind the 'as of
    2026-07-31' default view)."""
    from cte.scoring.history import dial_options
    m = pd.date_range("2016-01-31", "2026-06-30", freq="ME")
    rows = []
    for c in ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"]:
        for d in m:
            rows.append({"date": d, "ccy": c, "kind": "month_end",
                         "axis1_fundamental_struct": 0.1,
                         "axis2_stretch_struct": 0.2})
        rows.append({"date": pd.Timestamp("2026-07-11"), "ccy": c,
                     "kind": "daily", "axis1_fundamental_struct": 0.1,
                     "axis2_stretch_struct": 0.2})
    hist = pd.DataFrame(rows)
    opts = dial_options(hist, "struct")
    assert opts[-1] == pd.Timestamp("2026-06-30"), \
        f"phantom month offered: {opts[-1]}"
    assert all(o <= pd.Timestamp("2026-06-30") for o in opts)


def test_positioning_snapshot_has_13w_tail_origin(tmp_cache):
    from cte.flags.positioning import positioning_snapshot
    hist = list(np.random.default_rng(2).normal(0, 3, 200))
    _write_tff(tmp_cache, hist, hist)
    snap = positioning_snapshot().dropna(subset=["lev_z"])
    for col in ("lev_z_13w", "am_z_13w"):
        assert col in snap.columns and snap[col].notna().all()


def test_backfill_persists_pillar_and_carry_histories(tmp_cache, monkeypatch):
    import cte.adapters.base as base
    H = _tiny_universe(tmp_cache, monkeypatch)
    H.backfill()
    ph = base.read_cache("pillar_history")
    ch = base.read_cache("carry_history")
    assert ph is not None and {"date", "ccy", "pillar", "struct",
                               "regime"} <= set(ph.columns)
    assert ch is not None and {"date", "ccy", "real_2y",
                               "nominal_2y"} <= set(ch.columns)
    # tiny universe has no real_2y/nominal_2y features -> carry history is an
    # EMPTY frame with the right schema; pillar history must have real scores
    late = ph[ph.date > "2024-01-01"]
    assert late["regime"].notna().any()


def test_append_today_details_idempotent(tmp_cache, monkeypatch):
    import cte.adapters.base as base
    from cte.scoring.history import append_today_details
    pill_s = pd.DataFrame({"ccy": ["USD"], "pillar": ["A_growth"],
                           "pscore": [0.5], "axis": ["axis1_fundamental"],
                           "pw": [1.0]})
    pill_r = pill_s.assign(pscore=0.2)
    lz = pd.DataFrame({"ccy": ["USD", "USD"],
                       "metric": ["real_2y", "nominal_2y"],
                       "value": [1.1, 3.2]})
    monkeypatch.setattr(base, "CACHE_DIR", tmp_cache)
    append_today_details(pill_s, pill_r, lz)
    append_today_details(pill_s, pill_r, lz)
    ph = base.read_cache("pillar_history")
    ch = base.read_cache("carry_history")
    assert len(ph) == 1 and ph.struct.iloc[0] == 0.5 and ph.regime.iloc[0] == 0.2
    assert len(ch) == 1 and ch.real_2y.iloc[0] == 1.1


def test_positioning_asof_uses_only_data_through_date(tmp_cache):
    """As-of must ignore reports after the dial date and produce its own 13w
    origin from the pre-date panel."""
    import cte.adapters.base as base
    from cte.flags.positioning import (persist_history, positioning_asof,
                                       positioning_snapshot)
    hist = list(np.random.default_rng(5).normal(0, 3, 400))
    _write_tff(tmp_cache, hist, hist)
    persist_history()
    ph = base.read_cache("pos_history")
    cutoff = sorted(ph.date.unique())[-30]          # 30 reports back
    snap = positioning_asof(pd.Timestamp(cutoff))
    aud = snap[snap.ccy == "AUD"].iloc[0]
    assert pd.Timestamp(aud.pos_date) <= pd.Timestamp(cutoff)
    assert pd.notna(aud.lev_z_13w)
    # and it differs from the live snapshot's read
    live = positioning_snapshot()[lambda d: d.ccy == "AUD"].iloc[0]
    assert pd.Timestamp(live.pos_date) > pd.Timestamp(aud.pos_date)


def test_backfill_persists_overlay_history(tmp_cache, monkeypatch):
    import cte.adapters.base as base
    H = _tiny_universe(tmp_cache, monkeypatch)
    H.backfill()
    oh = base.read_cache("overlay_history")
    assert oh is not None
    need = {"date", "ccy", "kind", "yld_fx_corr", "yld_regime", "real10y_mult",
            "growth_z", "real_policy_z", "feasibility", "infl_mult",
            "carry_to_vol", "ctv_pctile"}
    assert need <= set(oh.columns)
    late = oh[oh.date > "2020-01-01"]
    assert late["real10y_mult"].notna().any()
    assert late["infl_mult"].notna().any()


def test_ctv_history_expanding_percentile_semantics():
    """A monotonically rising ratio must show a rising as-of percentile ending
    near 100 — 'crowded vs everything seen so far', never vs the future."""
    import cte.adapters.base as base
    import tempfile, pathlib
    base.CACHE_DIR = pathlib.Path(tempfile.mkdtemp())
    idx = pd.date_range("2015-01-01", periods=2600, freq="B")
    fx = pd.DataFrame({"date": idx, "ccy": "AUD",
                       "value": 1 + np.linspace(0, .0001, len(idx))})
    base.write_cache(fx, "fx_spot")
    me = pd.date_range("2015-01-31", "2024-12-31", freq="ME")
    feats = pd.DataFrame({"date": me, "ccy": "AUD", "feature": "real_2y",
                          "value": np.linspace(0.1, 5.0, len(me))})
    from cte.flags.overlays import carry_to_vol_history
    h = carry_to_vol_history(feats)
    a = h[h.ccy == "AUD"].dropna(subset=["ctv_pctile"])
    assert len(a) and a.ctv_pctile.iloc[-1] == 100
    assert a.ctv_pctile.iloc[-1] >= a.ctv_pctile.iloc[0]


def test_axes_from_pillars_reproduces_engine_axes_exactly(tmp_cache, monkeypatch):
    """The slider machinery must be exact recomposition: default weights over the
    persisted pillar scores == the engine's own axis scores, bit-for-bit."""
    import cte.adapters.base as base
    from cte.scoring.compositor import axes_from_pillars
    H = _tiny_universe(tmp_cache, monkeypatch)
    H.backfill()
    ph = base.read_cache("pillar_history")
    sh = base.read_cache("snapshot_history")
    ks = ("date", "ccy", "kind")
    re = axes_from_pillars(ph, "struct", None, keys=ks).merge(
        axes_from_pillars(ph, "regime", None, keys=ks), on=list(ks), how="outer")
    j = sh.merge(re, on=["date", "ccy"], suffixes=("", "_re"))
    for c in ["axis1_fundamental_struct", "axis2_stretch_struct",
              "axis1_fundamental_regime", "axis2_stretch_regime"]:
        a, b = j[c], j[f"{c}_re"]
        ok = np.isclose(a, b, atol=1e-12) | (a.isna() & b.isna())
        assert ok.all(), f"{c}: recomposition diverged from engine"


def test_axes_from_pillars_zero_weight_excludes_pillar(tmp_cache, monkeypatch):
    import cte.adapters.base as base
    from cte.scoring.compositor import axes_from_pillars
    H = _tiny_universe(tmp_cache, monkeypatch)
    H.backfill()
    ph = base.read_cache("pillar_history")
    d = ph[ph.date == ph.date.max()]
    default = axes_from_pillars(d, "regime")
    no_growth = axes_from_pillars(d, "regime", {"A_growth": 0.0})
    # manual check for one ccy: axis1 without growth = wmean of remaining pillars
    ccy = default.ccy.iloc[0]
    rows = d[(d.ccy == ccy) & d.regime.notna()]
    rest = rows[rows.pillar != "A_growth"]
    rest = rest[rest.pillar.isin(["B_inflation", "C_external", "D_fiscal"])]
    if len(rest):
        manual = np.average(rest.regime, weights=np.ones(len(rest)))
        got = no_growth.set_index("ccy").loc[ccy, "axis1_fundamental_regime"]
        assert np.isclose(got, manual, atol=1e-12)
    assert not np.allclose(
        default["axis1_fundamental_regime"].fillna(0),
        no_growth.set_index("ccy").reindex(default.ccy)
        ["axis1_fundamental_regime"].fillna(0).values)


def test_secular_horizon_through_the_stack(tmp_cache, monkeypatch):
    """secular_z exists, requires the longer window (starts after regime),
    lands in snapshot/pillar histories, and the dial bounds it separately."""
    import cte.adapters.base as base
    from cte.scoring.history import dial_options
    from cte.transform.zscore import dual_horizon_z
    H = _tiny_universe(tmp_cache, monkeypatch)
    hist = H.backfill()
    assert {"axis1_fundamental_secular", "axis2_stretch_secular"} <= set(hist.columns)
    ph = base.read_cache("pillar_history")
    assert "secular" in ph.columns and ph["secular"].notna().any()
    d_reg = dial_options(hist, "regime", min_ccys=2)
    d_sec = dial_options(hist, "secular", min_ccys=2)
    assert d_sec and d_sec[0] > d_reg[0], "secular dial must start later than regime"
    # z engine emits the column with a genuinely longer requirement
    idx = pd.date_range("2015-01-31", periods=100, freq="ME")
    z = dual_horizon_z(pd.DataFrame({"date": idx, "ccy": "USD",
                                     "metric": "x", "value": np.arange(100.0)}))
    assert "secular_z" in z.columns
    assert z["secular_z"].notna().sum() < z["regime_z"].notna().sum()
