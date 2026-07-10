"""Regression tests for the narrative-notes module and the completeness guard —
covering the whole-pillar-outage bug (Fable review #1) and the module's KeyError edge."""
import numpy as np
import pandas as pd

from cte.flags.notes import completeness_warnings, context_notes


def _tm(**kw):
    base = {"ccy": ["XXX"], "axis1_fundamental_struct": [0.0],
            "axis2_stretch_struct": [0.0], "axis1_fundamental_regime": [0.0],
            "axis2_stretch_regime": [0.0]}
    base.update({k: [v] for k, v in kw.items()})
    return pd.DataFrame(base)


def _pillars(**kw):
    row = {"A_growth": 0.1, "B_inflation": 0.1, "C_external": 0.1,
           "D_fiscal": 0.1, "E_policy": 0.1, "G_valuation": 0.1}
    row.update(kw)
    return pd.DataFrame(row, index=["XXX"])


# ---- completeness guard: the whole-pillar-outage bug (review #1) ----

def test_completeness_catches_pillar_missing_for_all_currencies():
    # Pivot with D_fiscal and E_policy absent for EVERY currency (a source outage).
    pillars = pd.DataFrame(
        {"A_growth": [0.1, -0.2], "B_inflation": [0.0, 0.1],
         "C_external": [0.3, -0.1], "G_valuation": [0.5, -0.3]},
        index=["USD", "EUR"])
    out = completeness_warnings(pillars)
    assert set(out) == {"USD", "EUR"}
    assert "Real 10Y" in out["USD"] and "Policy" in out["USD"]


def test_completeness_silent_when_all_pillars_present():
    assert completeness_warnings(_pillars()) == {}


# ---- narrative notes ----

def test_twin_signal_cheap_and_improving():
    notes = context_notes(_tm(axis1_fundamental_struct=0.6, axis2_stretch_struct=-0.6),
                          _pillars(G_valuation=-0.6),
                          pd.DataFrame({"XXX": [np.nan]}, index=["XXX"]))
    assert any("Cheap and improving" in n for n in notes["XXX"])


def test_regime_divergence_off_its_highs():
    notes = context_notes(_tm(axis2_stretch_struct=0.7, axis2_stretch_regime=-0.5),
                          _pillars(), pd.DataFrame({"XXX": [np.nan]}, index=["XXX"]))
    assert any("off its highs" in n for n in notes["XXX"])


def test_one_legged_read_flags_single_dominant_pillar():
    notes = context_notes(_tm(), _pillars(G_valuation=2.7),
                          pd.DataFrame({"XXX": [np.nan]}, index=["XXX"]))
    assert any("One-legged read" in n and "Valuation" in n for n in notes["XXX"])


def test_notes_do_not_raise_when_regime_columns_absent():
    # thin cache: regime columns missing entirely -> must degrade, not KeyError
    tm = pd.DataFrame({"ccy": ["XXX"], "axis1_fundamental_struct": [0.6],
                       "axis2_stretch_struct": [-0.6]})
    notes = context_notes(tm, _pillars(G_valuation=-0.6),
                          pd.DataFrame({"XXX": [np.nan]}, index=["XXX"]))
    assert isinstance(notes, dict)  # twin-signal still fires; no crash on missing regime
