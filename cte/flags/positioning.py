"""Positioning overlay (flag layer) — CFTC TFF futures-and-options-combined.

Scope, stated honestly: this is the LISTED-derivatives slice (CME/ICE futures plus
delta-adjusted options), i.e. speculative futures positioning — leveraged funds (fast
money) and asset managers (real money). It is NOT holistic FX positioning: OTC
forwards/swaps, corporate hedging, and real-money cash flows are invisible here. It is
used strictly as a crowding / unwind-risk conditioner, the population for which the
futures slice is a defensible proxy — and per spec §11 it is an overlay, never part of
the axis composites: positioning doesn't change what a currency is worth, it changes
the risk of holding the consensus side of it.

Conventions:
  - Net positions are normalized as % of open interest so the z-score is comparable
    across two decades of changing participation.
  - Sign is per-currency: every contract is quoted as the currency vs USD (long =
    long the currency); the USD row uses the ICE Dollar Index (long = long USD),
    matching the FX block's convention.
  - z-scores use the shared calendar-window engine (_roll_z) on the struct horizon;
    a 13-week change z captures the *swing* in positioning, not just the level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cte.adapters.base import read_cache
from cte.config import (CURRENCIES, POS_CROWDED_Z, POS_DIVERGE_Z,
                        POS_STRUCT_YEARS)
from cte.transform.zscore import _roll_z


def _panel() -> pd.DataFrame | None:
    tff = read_cache("tff")
    if tff is None or tff.empty:
        return None
    tff = tff.dropna(subset=["date"]).sort_values(["ccy", "date"])
    # derive am %OI for rows cached before the adapter emitted it
    if "am_net_pct_oi" not in tff.columns:
        tff["am_net_pct_oi"] = np.where(
            tff["open_interest"].fillna(0) > 0,
            100 * tff["am_net"] / tff["open_interest"], np.nan)
    return tff


def positioning_history() -> pd.DataFrame:
    """Weekly per-ccy panel: lev/am net %OI plus their level-z and lev 13w-change-z.
    Empty frame if the tff cache is absent (positioning then degrades gracefully
    everywhere downstream)."""
    tff = _panel()
    if tff is None:
        return pd.DataFrame(columns=["date", "ccy", "lev_pct_oi", "am_pct_oi",
                                     "lev_z", "am_z", "lev_chg13w_z"])
    rows = []
    for ccy, g in tff.groupby("ccy"):
        g = g.set_index("date").sort_index()
        lev = g["lev_net_pct_oi"].astype(float).dropna()
        am = g["am_net_pct_oi"].astype(float).dropna()
        if len(lev) < 2:
            continue
        d = pd.DataFrame({
            "lev_pct_oi": lev,
            "am_pct_oi": am,
            "lev_z": _roll_z(lev, POS_STRUCT_YEARS),
            "am_z": _roll_z(am, POS_STRUCT_YEARS) if len(am) >= 2 else np.nan,
            "lev_chg13w_z": _roll_z(lev.diff(13).dropna(), POS_STRUCT_YEARS)
                            if len(lev) > 13 else np.nan,
        })
        d["ccy"] = ccy
        rows.append(d.reset_index().rename(columns={"index": "date"}))
    if not rows:
        return pd.DataFrame(columns=["date", "ccy", "lev_pct_oi", "am_pct_oi",
                                     "lev_z", "am_z", "lev_chg13w_z"])
    return pd.concat(rows, ignore_index=True)


def _label(lev_z: float, am_z: float) -> str:
    if pd.isna(lev_z):
        return "no data"
    if abs(lev_z) >= POS_CROWDED_Z:
        side = "long" if lev_z > 0 else "short"
        tag = f"CROWDED {side}"
        if pd.notna(am_z) and abs(am_z) >= POS_DIVERGE_Z and \
                np.sign(am_z) != np.sign(lev_z):
            tag += " / real-money opposed"
        return tag
    if pd.notna(am_z) and abs(am_z) >= POS_DIVERGE_Z and \
            abs(lev_z) >= POS_DIVERGE_Z and np.sign(am_z) != np.sign(lev_z):
        return "spec vs real-money split"
    return "normal"


def positioning_snapshot() -> pd.DataFrame:
    """Latest positioning read per currency:
      [ccy, lev_pct_oi, am_pct_oi, lev_z, am_z, lev_chg13w_z, pos_label, pos_date]
    Merged into overlay_snapshot so it persists with the committed overlays."""
    hist = positioning_history()
    if hist.empty:
        return pd.DataFrame({"ccy": list(CURRENCIES)})
    last = (hist.sort_values("date").groupby("ccy").tail(1)
            .rename(columns={"date": "pos_date"}))
    last["pos_label"] = [
        _label(r.lev_z, r.am_z) for r in last.itertuples()]
    cols = ["ccy", "lev_pct_oi", "am_pct_oi", "lev_z", "am_z",
            "lev_chg13w_z", "pos_label", "pos_date"]
    out = last[cols].round({"lev_pct_oi": 1, "am_pct_oi": 1, "lev_z": 2,
                            "am_z": 2, "lev_chg13w_z": 2})
    return out.set_index("ccy").reindex(CURRENCIES).reset_index()


def positioning_warnings(snap: pd.DataFrame,
                         tm: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Crowding / divergence sidenotes, plus the three-way setup when the tension
    map is supplied: the map's vulnerable quadrant (rich + deteriorating) with
    stretched LONG spec positioning is the dangerous combination — a fragile
    position with a catalyst attached. (Symmetrically for washed-out + crowded
    short.) Positioning never bends the axis scores; it only speaks here."""
    out: dict[str, list[str]] = {}
    if snap is None or "lev_z" not in snap.columns:
        return out
    axes = None
    if tm is not None and "axis1_fundamental_struct" in tm.columns:
        axes = tm.set_index("ccy")
    for _, r in snap.iterrows():
        c, notes = r["ccy"], []
        lev_z = r.get("lev_z", np.nan)
        if pd.isna(lev_z):
            continue
        label = str(r.get("pos_label", ""))
        if label.startswith("CROWDED"):
            side = "long" if lev_z > 0 else "short"
            notes.append(
                f"Positioning crowded: leveraged funds are {side} at a "
                f"{lev_z:+.1f}z extreme ({r.get('lev_pct_oi', float('nan')):+.0f}% of "
                f"open interest, futures+options). Consensus side of the boat — "
                f"unwind risk is elevated if the tape turns. (Listed-derivatives "
                f"slice only; OTC positioning is not visible here.)")
        if "real-money opposed" in label or label == "spec vs real-money split":
            notes.append(
                f"Spec vs real money split: leveraged funds ({lev_z:+.1f}z) and "
                f"asset managers ({r.get('am_z', float('nan')):+.1f}z) are on opposite "
                f"sides — fast money is leaning against the institutional stance, a "
                f"setup that resolves sharply when one side capitulates.")
        # three-way: quadrant + crowding coincide
        if axes is not None and c in axes.index:
            f = axes.loc[c].get("axis1_fundamental_struct", np.nan)
            v = axes.loc[c].get("axis2_stretch_struct", np.nan)
            if pd.notna(f) and pd.notna(v):
                if f < -0.25 and v > 0.25 and lev_z >= POS_CROWDED_Z:
                    notes.append(
                        f"Vulnerable AND crowded long: rich valuation ({v:+.1f}), "
                        f"deteriorating fundamentals ({f:+.1f}), and spec money "
                        f"stretched long ({lev_z:+.1f}z) — the map's dangerous "
                        f"quadrant with a catalyst attached. Classic unwind setup.")
                elif f > 0.25 and v < -0.25 and lev_z <= -POS_CROWDED_Z:
                    notes.append(
                        f"Washed out AND crowded short: cheap ({v:+.1f}), improving "
                        f"({f:+.1f}), and spec money stretched short ({lev_z:+.1f}z) "
                        f"— the squeeze setup: any positive surprise forces covering "
                        f"into an improving story.")
        if notes:
            out[c] = notes
    return out


if __name__ == "__main__":
    snap = positioning_snapshot()
    print(snap.to_string(index=False))
    for c, notes in positioning_warnings(snap).items():
        print(f"\n{c}:")
        for n in notes:
            print(f"  - {n}")
