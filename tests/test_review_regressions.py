"""Regression tests for the July 2026 code-review findings (F1-F6).

Each test encodes the *intended* behavior, so the tests for unfixed findings
FAIL today and pass once the fix lands. Run:

    pytest tests/test_review_regressions.py -v            # offline tests only
    pytest tests/test_review_regressions.py -v -m network # + live-endpoint checks

No API keys required. Network-marked tests hit only keyless endpoints (OECD SDMX,
BoE) and exist to pin the empirical facts the fixes rely on.

Finding map:
  F1  external.fetch_niip: missing FS_ENTRY=="LE" filter -> 5 rows/quarter for
      AUS/CAN/NZL, later averaged by features' pivot_table (wrong NIIP).
  F2  compositor._apply_overlays: inflation gate tests each row's own value, not
      the currency's infl_gap (momentum bent below target / not bent above).
  F3  features._chg/_pct: <2 observations returns raw LEVELS under a change name.
  F4  features._chg/_pct: positional lag on gappy/mixed-cadence series shifts the
      change window silently (12m becomes 13m across a missing month, etc.).
  F5  zscore._roll_z: single-observation series raises ValueError (int(NaN)).
  F6  Daily-CI persistence contradiction: fetch_boe(history=False) is a ~1-month
      tail, and the raw cache is neither committed nor cached between runs.
      (Unit-testable slice: the crash path F5 inside dual_horizon_z, plus a
      network check documenting the BoE tail depth; the workflow fix itself is
      asserted by a repo-hygiene test.)

Also included: lock-in tests for invariants verified correct during review
(z-window is strictly trailing; carry grid antisymmetry; merge_cache idempotency
and keep-newest revision handling) so future changes can't silently break them.
"""
from __future__ import annotations

import io
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cte.scoring.compositor import _apply_overlays              # noqa: E402
from cte.transform.features import _chg, _pct                   # noqa: E402
from cte.transform.zscore import _roll_z, dual_horizon_z        # noqa: E402


# --------------------------------------------------------------------------
# F1 — NIIP: FS_ENTRY filter + one-row-per-(ccy, quarter) contract
# --------------------------------------------------------------------------

_IIP_CSV = """\
MEASURE,ACCOUNTING_ENTRY,UNIT_MEASURE,FREQ,REF_AREA,FS_ENTRY,TIME_PERIOD,OBS_VALUE,UNIT_MULT
FA,N,XDC,Q,AUS,LE,2025-Q4,-700000,6
FA,N,XDC,Q,AUS,K,2025-Q4,-43000,6
FA,N,XDC,Q,AUS,K7A,2025-Q4,-42000,6
FA,N,XDC,Q,AUS,K7B,2025-Q4,-12000,6
FA,N,XDC,Q,AUS,KA,2025-Q4,11000,6
FA,N,XDC,Q,AUS,LE,2025-Q1,-690000,6
FA,N,XDC,Q,AUS,K,2025-Q1,-40000,6
FA,N,XDC,Q,AUS,LE,2025-Q2,-695000,6
FA,N,XDC,Q,AUS,LE,2025-Q3,-698000,6
FA,N,XDC,Q,USA,LE,2025-Q4,-27000000,6
FA,N,XDC,Q,USA,LE,2025-Q3,-26500000,6
FA,N,XDC,Q,USA,LE,2025-Q2,-26000000,6
FA,N,XDC,Q,USA,LE,2025-Q1,-25500000,6
"""

_QNA_CSV = """\
TRANSACTION,PRICE_BASE,ADJUSTMENT,FREQ,EXPENDITURE,REF_AREA,TIME_PERIOD,OBS_VALUE,UNIT_MULT
B1GQ,V,Y,Q,_Z,AUS,2025-Q1,700000,6
B1GQ,V,Y,Q,_Z,AUS,2025-Q2,710000,6
B1GQ,V,Y,Q,_Z,AUS,2025-Q3,720000,6
B1GQ,V,Y,Q,_Z,AUS,2025-Q4,730000,6
B1GQ,V,Y,Q,_Z,USA,2025-Q1,7800000,6
B1GQ,V,Y,Q,_Z,USA,2025-Q2,7850000,6
B1GQ,V,Y,Q,_Z,USA,2025-Q3,7900000,6
B1GQ,V,Y,Q,_Z,USA,2025-Q4,7950000,6
"""

_EUROSTAT_NIIP_CSV = """\
geo,TIME_PERIOD,OBS_VALUE
EA20,2025-Q4,-4.2
EA20,2025-Q3,-4.5
"""


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):  # pragma: no cover - never errors
        pass


class _FakeSession:
    """Serves canned SDMX-CSV by URL substring; no network."""

    def get(self, url, **kw):
        if "DF_IIP" in url:
            return _FakeResp(_IIP_CSV.encode())
        if "DF_QNA" in url:
            return _FakeResp(_QNA_CSV.encode())
        if "eurostat" in url or "tipsii40" in url:
            return _FakeResp(_EUROSTAT_NIIP_CSV.encode())
        raise AssertionError(f"unexpected URL in test: {url}")


def test_f1_niip_one_row_per_ccy_quarter_and_stock_only():
    """fetch_niip must keep only the closing-stock entry (FS_ENTRY == 'LE').

    FAILS pre-fix: the AUS 2025-Q4 quarter yields 5 rows (LE + K/K7A/K7B/KA),
    which features.py's pivot_table later averages into a wrong NIIP.
    """
    from cte.adapters import external

    out = external.fetch_niip(session=_FakeSession())
    aus = out[(out.ccy == "AUD")]

    # exactly one niip row per (ccy, quarter)
    dup = out.groupby(["ccy", "date"]).size()
    assert (dup == 1).all(), (
        f"duplicate niip rows per (ccy, date):\n{dup[dup > 1]}\n"
        "-> add `& (iip.FS_ENTRY == 'LE')` to the IIP filter in external.py"
    )

    # and the value must be the LE stock over trailing-4Q nominal GDP
    q4 = aus[aus.date == pd.Timestamp("2025-12-31")]
    assert len(q4) == 1
    expected = -700000 / (700000 + 710000 + 720000 + 730000) * 100.0
    assert q4.value.iloc[0] == pytest.approx(expected, abs=1e-6), (
        f"AUD 2025-Q4 NIIP%GDP = {q4.value.iloc[0]:.2f}, expected {expected:.2f} "
        "(the LE stock, not a mean over revaluation/flow entries)"
    )


@pytest.mark.network
def test_f1_live_oecd_iip_filter_is_unique_per_period():
    """Pin the live fact: the code's IIP filter without FS_ENTRY is non-unique.

    Passes once external.py pins FS_ENTRY == 'LE' (re-expressed here); if OECD
    ever restructures the dataflow this test flags it before the pipeline does.
    """
    import requests

    from cte.config import (HTTP_UA, OECD_CSV_ACCEPT, OECD_IIP_DATAFLOW,
                            OECD_SDMX_BASE)

    url = (f"{OECD_SDMX_BASE}/{OECD_IIP_DATAFLOW}/all"
           f"?lastNObservations=1&dimensionAtObservation=AllDimensions")
    r = requests.get(url, headers={"User-Agent": HTTP_UA,
                                   "Accept": OECD_CSV_ACCEPT}, timeout=120)
    r.raise_for_status()
    raw = pd.read_csv(io.BytesIO(r.content))
    f = raw[(raw.MEASURE == "FA") & (raw.ACCOUNTING_ENTRY == "N")
            & (raw.UNIT_MEASURE == "XDC") & (raw.FREQ == "Q")
            & (raw.FS_ENTRY == "LE")]
    sizes = f.groupby(["REF_AREA", "TIME_PERIOD"]).size()
    assert (sizes == 1).all(), f"non-unique after LE pin:\n{sizes[sizes > 1]}"
    # units must stay matched with the QNA denominator (both were 6 at review)
    assert set(f.UNIT_MULT.dropna().unique()) == {6}


# --------------------------------------------------------------------------
# F2 — inflation overlay gate: keyed on the currency's infl_gap, not row value
# --------------------------------------------------------------------------

def _snap(infl_mult: float) -> pd.DataFrame:
    return pd.DataFrame({"ccy": ["XXX"], "real10y_mult": [np.nan],
                         "infl_mult": [infl_mult]})


def test_f2_below_target_bends_neither_inflation_row():
    """Below-target currency: trap logic must NOT bend infl_momentum.

    FAILS pre-fix: momentum row is multiplied whenever its own value > 0.
    """
    d = pd.DataFrame({
        "ccy": ["XXX", "XXX"],
        "metric": ["infl_gap", "infl_momentum"],
        "value": [-0.5, +0.8],          # below target, but momentum rising
        "signed": [1.0, 1.0],
    })
    out = _apply_overlays(d, _snap(-0.7))
    mom = out.loc[out.metric == "infl_momentum", "signed"].iloc[0]
    assert mom == pytest.approx(1.0), (
        f"infl_momentum bent to {mom} while inflation is BELOW target — the "
        "gate must test the currency's infl_gap, not the row's own value"
    )


def test_f2_above_target_bends_both_inflation_rows():
    """Above-target currency: the trap multiplier applies to the whole
    inflation pillar, including decelerating momentum.

    FAILS pre-fix: a negative momentum row escapes the gate.
    """
    d = pd.DataFrame({
        "ccy": ["XXX", "XXX"],
        "metric": ["infl_gap", "infl_momentum"],
        "value": [+0.5, -0.8],          # above target, momentum falling
        "signed": [1.0, 1.0],
    })
    out = _apply_overlays(d, _snap(-0.7))
    gap = out.loc[out.metric == "infl_gap", "signed"].iloc[0]
    mom = out.loc[out.metric == "infl_momentum", "signed"].iloc[0]
    assert gap == pytest.approx(-0.7)
    assert mom == pytest.approx(-0.7), (
        f"infl_momentum left at {mom} while inflation is ABOVE target — "
        "REVIEW.md #4 says the gate is above-target, per currency"
    )


# --------------------------------------------------------------------------
# F3 — _chg/_pct must never emit levels under a change name
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [_chg, _pct], ids=["_chg", "_pct"])
def test_f3_short_series_returns_no_values(fn):
    """FAILS pre-fix: a 1-obs series is returned as-is (the raw level)."""
    one = pd.Series([21000.0], index=pd.DatetimeIndex(["2026-03-31"]))
    out = fn(one, 12)
    assert out.dropna().empty, (
        f"{fn.__name__} returned {out.dropna().tolist()} for a single-"
        "observation series — that is the raw LEVEL, not a change"
    )


# --------------------------------------------------------------------------
# F4 — calendar-correct lags on gappy / mixed-cadence series
# --------------------------------------------------------------------------

def test_f4_missing_month_does_not_shift_the_yoy_window():
    """Monthly series with one dropped month: the '12-month' pct change taken
    at the latest observation must still span ~12 calendar months.

    FAILS pre-fix: the positional lag lands 13 months back past the gap.
    """
    idx = pd.date_range("2020-01-31", "2026-06-30", freq="ME")
    s = pd.Series(100.0 * 1.002 ** np.arange(len(idx)), index=idx)
    s = s.drop(pd.Timestamp("2026-01-31"))  # one suppressed print, recent
    out = _pct(s, 12)
    # assert at the latest obs: its 12-row positional lag now crosses the gap,
    # landing 13 calendar months back
    true_12m = (s.loc["2026-06-30"] / s.loc["2025-06-30"] - 1) * 100
    assert out.loc["2026-06-30"] == pytest.approx(true_12m, abs=1e-9), (
        f"12m pct at 2026-06-30 = {out.loc['2026-06-30']:.4f}, true calendar "
        f"12m = {true_12m:.4f} — positional lag shifted by the missing month"
    )


def test_f4_mixed_cadence_3m_change_on_the_monthly_tail():
    """Quarterly history + monthly tail (the cadence-switch / orphan-row case):
    the '3-month' change at the latest monthly observation must span ~3 months.

    FAILS pre-fix: median-gap inference picks lag=1 row (a 1-month change).
    """
    q = pd.date_range("2018-03-31", "2025-09-30", freq="QE")
    m = pd.date_range("2025-10-31", "2026-06-30", freq="ME")
    vals = list(np.arange(len(q), dtype=float)) + \
           list(np.arange(1, len(m) + 1) / 3 + len(q) - 1)
    s = pd.Series(vals, index=q.append(m))
    out = _chg(s, 3)
    true_3m = s.iloc[-1] - s.loc["2026-03-31"]
    assert out.iloc[-1] == pytest.approx(true_3m, abs=1e-9), (
        f"3m change on the monthly tail = {out.iloc[-1]:.4f}, true = "
        f"{true_3m:.4f} — mixed cadence corrupted the lag inference"
    )


# --------------------------------------------------------------------------
# F5 / F6(crash path) — z-score layer must survive degenerate series
# --------------------------------------------------------------------------

def test_f5_roll_z_single_observation_returns_nan_not_crash():
    """FAILS pre-fix: ValueError (int(NaN)) from the min_periods inference."""
    s = pd.Series([1.0], index=pd.DatetimeIndex(["2024-06-30"]))
    out = _roll_z(s, 10)          # must not raise
    assert out.isna().all()


def test_f5_dual_horizon_z_tolerates_one_short_group():
    """One 1-obs (ccy, metric) group must not take down the whole panel —
    this is exactly the GBP state a fresh CI runner produces (F6).

    FAILS pre-fix with ValueError.
    """
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    good = pd.DataFrame({"date": idx, "ccy": "USD", "metric": "nominal_2y",
                         "value": np.linspace(1, 4, len(idx))})
    stub = pd.DataFrame({"date": [pd.Timestamp("2026-06-30")], "ccy": "GBP",
                         "metric": "nominal_2y", "value": [3.9]})
    z = dual_horizon_z(pd.concat([good, stub], ignore_index=True))
    gbp = z[z.ccy == "GBP"]
    assert len(gbp) == 1
    assert gbp[["struct_z", "regime_z"]].isna().all().all()
    # the healthy group must still score
    assert z[z.ccy == "USD"]["struct_z"].notna().any()


# --------------------------------------------------------------------------
# F6 — repo hygiene: the daily Action must be able to produce full history
# --------------------------------------------------------------------------

def test_f6_daily_workflow_has_a_persistence_or_full_history_story():
    """The daily job runs on a fresh checkout with the raw caches gitignored.
    Until the workflow either (a) persists the raw caches (actions/cache,
    artifacts, or committing them) or (b) stops passing --daily so every
    adapter pulls full history (GBP's fetch_boe(history=False) is a ~1-month
    tail), GBP's z-scored pillars cannot be built in CI.

    FAILS pre-fix. Satisfy it by whichever route you pick; loosen the check
    accordingly if you choose a mechanism it doesn't recognize.
    """
    wf = (REPO / ".github" / "workflows" / "refresh.yml").read_text()
    gitignore = (REPO / ".gitignore").read_text()

    caches_ignored = "data/cache/yields.parquet" in gitignore
    uses_actions_cache = "actions/cache" in wf
    uploads_artifact = "upload-artifact" in wf and "download-artifact" in wf
    commits_raw_cache = "yields.parquet" in wf
    runs_full_history = ("scripts.backfill --daily" not in wf
                         and "scripts.backfill" in wf)

    persisted = (not caches_ignored) or uses_actions_cache or \
        uploads_artifact or commits_raw_cache
    assert persisted or runs_full_history, (
        "raw caches are gitignored, the workflow neither restores nor persists "
        "them, and backfill runs with --daily — a fresh runner rebuilds GBP "
        "yields from the current-month BoE tail only (fetch_boe(history=False))"
    )


def test_f6_narrator_prev_snapshot_persists_if_change_section_is_wanted():
    """tension_map_prev.parquet is written by the narrator for day-over-day
    change detection but is gitignored and never committed by the workflow,
    so the 'CHANGE SINCE PRIOR SNAPSHOT' section can never fire in CI.

    FAILS pre-fix. Either commit it in the workflow or drop the feature.
    """
    wf = (REPO / ".github" / "workflows" / "refresh.yml").read_text()
    gitignore = (REPO / ".gitignore").read_text()
    persisted = ("tension_map_prev" in wf) or \
                ("tension_map_prev" not in gitignore)
    assert persisted, (
        "tension_map_prev.parquet never survives between ephemeral runners — "
        "the narrator's change section is dead code in production"
    )


@pytest.mark.network
def test_f6_boe_latest_zip_is_a_short_tail():
    """Documents the empirical constraint F6 rests on: the BoE 'latest' zip is
    a current-month tail, not history. If BoE ever changes this, the finding
    (and the workflow requirement) should be revisited."""
    from cte.adapters.sovereign_yields import fetch_boe

    df = fetch_boe(history=False)
    span_days = (df.date.max() - df.date.min()).days
    assert span_days < 62, (
        f"BoE latest zip now spans {span_days} days — F6's premise changed"
    )


# --------------------------------------------------------------------------
# Lock-ins — invariants verified correct at review time
# --------------------------------------------------------------------------

def test_lockin_zscore_window_is_strictly_trailing():
    idx = pd.date_range("2010-01-31", periods=180, freq="ME")
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=180), index=idx)
    z1 = _roll_z(s, 10)
    s2 = s.copy()
    s2.iloc[-1] = 1e6                      # perturb ONLY the final observation
    z2 = _roll_z(s2, 10)
    pd.testing.assert_series_equal(z1.iloc[:-1], z2.iloc[:-1])


def test_lockin_carry_grid_antisymmetry(monkeypatch):
    import cte.transform.pairwise as pw

    feats = pd.DataFrame({
        "date": pd.Timestamp("2026-06-30"),
        "ccy": ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"],
        "feature": "real_2y",
        "value": [1.9, 0.4, -1.2, 1.1, -0.3, 0.8, 1.5, 1.6],
    })
    monkeypatch.setattr(pw, "build_features", lambda: feats)
    g = pw.carry_grid("real_2y")
    assert np.allclose(g.values, -g.values.T, atol=1e-9)
    assert np.allclose(np.diag(g.values), 0.0)


def test_lockin_merge_cache_idempotent_and_keeps_newest(tmp_path, monkeypatch):
    import cte.adapters.base as base
    import cte.store as store

    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)

    df = pd.DataFrame({"ccy": ["AUD"] * 3, "metric": ["cpi"] * 3,
                       "date": pd.to_datetime(["2025-03-31", "2025-06-30",
                                               "2025-09-30"]),
                       "value": [3.0, 3.2, 3.4],
                       "fetched_at": pd.Timestamp("2026-01-01")})
    m1 = store.merge_cache("t", df, keys=["ccy", "metric", "date"])
    m2 = store.merge_cache("t", df, keys=["ccy", "metric", "date"])
    pd.testing.assert_frame_equal(m1, m2)

    rev = df.copy()
    rev["value"] = [3.0, 3.2, 9.9]
    rev["fetched_at"] = pd.Timestamp("2026-02-01")
    m3 = store.merge_cache("t", rev, keys=["ccy", "metric", "date"])
    assert m3.value.tolist() == [3.0, 3.2, 9.9]
    assert len(m3) == 3
