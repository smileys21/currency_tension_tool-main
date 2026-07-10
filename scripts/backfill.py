"""One-time cold-start backfill + depth audit.

Pulls every ingestion block at full available history, persists to the parquet
cache via idempotent merge, and prints a z-score depth audit so we can see which
series actually support the ~10y structural and ~2y regime windows before scoring.

Run once to seed baselines:   python -m scripts.backfill
The daily GitHub Action re-runs the same merges to append new observations.

Env: FRED_API_KEY, ESTAT_APP_ID.
"""
from __future__ import annotations

import sys

import pandas as pd

from cte.store import depth_audit, merge_cache


def _try(label, fn):
    try:
        df = fn()
        print(f"  [ok]   {label:16} rows={len(df):>7}")
        return df
    except Exception as e:
        print(f"  [FAIL] {label:16} {type(e).__name__}: {str(e)[:60]}")
        return None


def run(full_history: bool = True) -> None:
    mode = "cold-start (full history)" if full_history else "daily incremental"
    print(f"Backfilling CTE cache — {mode}...\n")

    # --- fundamental macro panel (bcicp, cpi, unemp, gdp, current_account, policy, real_10y)
    from cte.adapters.macro import build_macro_backbone
    macro = _try("macro_backbone", build_macro_backbone)
    if macro is not None:
        merge_cache("macro_backbone", macro, keys=["ccy", "metric", "date"])

    # --- market blocks
    from cte.adapters import bis_reer, cftc_tff, sovereign_yields, yahoo
    fx = _try("fx_spot", lambda: yahoo.fetch_fx_spot(period="10y"))
    if fx is not None:
        merge_cache("fx_spot", fx, keys=["ccy", "date"])

    yields = _try("yields", lambda: sovereign_yields.fetch_all_yields(
        full_history=full_history))
    if yields is not None:
        merge_cache("yields", yields, keys=["ccy", "tenor", "date"])

    reer = _try("reer", bis_reer.fetch_reer)
    if reer is not None:
        merge_cache("reer", reer, keys=["ccy", "date"])

    tff = _try("tff", lambda: cftc_tff.fetch_tff(weeks_back=850))
    if tff is not None:
        merge_cache("tff", tff, keys=["ccy", "date"])

    # --- depth audit on the scored inputs
    print("\nDepth audit (macro backbone):")
    if macro is not None:
        aud = depth_audit(macro, ["ccy", "metric"])
        short = aud[aud.struct_10y == "SHORT"]
        print(aud.to_string(index=False))
        print(f"\n  {len(aud)} series | {len(short)} short of the 10y structural window")
        if len(short):
            print("  structural-short series:",
                  ", ".join(sorted(short.metric.unique())))

    for name, gcols in [("reer", ["ccy"]), ("tff", ["ccy"]),
                        ("yields", ["ccy", "tenor"])]:
        from cte.adapters.base import read_cache
        d = read_cache(name)
        if d is not None and "date" in d.columns:
            a = depth_audit(d, gcols)
            print(f"\nDepth audit ({name}): "
                  f"median span {a.years.median():.1f}y, "
                  f"{(a.struct_10y=='SHORT').sum()}/{len(a)} short of 10y")


if __name__ == "__main__":
    # cold-start by default; `--daily` does the lighter incremental merge
    run(full_history="--daily" not in sys.argv)
    sys.exit(0)
