"""Verify every sovereign-yield adapter against a live pull.

Prints, per source: rows returned, tenors present, the carry leg chosen by the
2Y->3Y fallback, and the latest 2Y / 10Y observations with their dates.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from cte.adapters import sovereign_yields as sy

pd.set_option("display.width", 160)

def latest(df, ccy, tenor):
    sub = df[(df.ccy == ccy) & (df.tenor == tenor)].dropna(subset=["value"])
    if sub.empty:
        return None, None
    row = sub.sort_values("date").iloc[-1]
    return row["value"], row["date"].date()

# pull primary sources individually so one failure doesn't mask the rest
sources = [
    ("USD", "us_treasury", sy.fetch_us),
    ("EUR", "ecb_yc",      sy.fetch_ecb),
    ("EUR_DE","bundesbank",sy.fetch_bundesbank),
    ("JPY", "mof_jgb",     sy.fetch_jp),
    ("GBP", "boe_glc",     sy.fetch_boe),
    ("CHF", "snb_spot",    sy.fetch_snb),
    ("CAD", "boc_valet",   sy.fetch_boc),
    ("AUD", "rba_f2",      sy.fetch_rba),
    ("NZD", "rbnz_b2",     sy.fetch_rbnz),
]

print(f"{'ccy':<6}{'source':<13}{'rows':>6}  {'tenors':<14}{'carry':<7}"
      f"{'2Y(date)':<22}{'10Y(date)':<22}{'status'}")
print("-" * 108)

all_frames = []
for ccy, label, fn in sources:
    t0 = time.time()
    try:
        df = fn()
        all_frames.append(df)
        ten = sorted(df["tenor"].unique(), key=lambda x: int(x[:-1]))
        carry = sy.resolve_carry_tenor(df[df.ccy == ccy]) if ccy != "EUR_DE" else \
                sy.resolve_carry_tenor(df)
        v2, d2 = latest(df, df.ccy.iloc[0], "2Y")
        v10, d10 = latest(df, df.ccy.iloc[0], "10Y")
        s2 = f"{v2:.3f} ({d2})" if v2 is not None else "—"
        s10 = f"{v10:.3f} ({d10})" if v10 is not None else "—"
        dt = time.time() - t0
        print(f"{ccy:<6}{label:<13}{len(df):>6}  {','.join(ten):<14}{carry or '—':<7}"
              f"{s2:<22}{s10:<22}ok {dt:4.1f}s")
    except Exception as e:
        dt = time.time() - t0
        print(f"{ccy:<6}{label:<13}{'—':>6}  {'—':<14}{'—':<7}{'—':<22}{'—':<22}"
              f"FAIL {type(e).__name__}: {e}")

print("-" * 108)
if all_frames:
    full = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows across all sources: {len(full):,}")
    print("Per-currency latest-date freshness:")
    for ccy in ["USD","EUR","JPY","GBP","CHF","CAD","AUD","NZD"]:
        sub = full[full.ccy == ccy]
        if not sub.empty:
            print(f"  {ccy}: latest obs {sub['date'].max().date()}  "
                  f"({sub['date'].min().date()} → {sub['date'].max().date()}, "
                  f"{sub['date'].nunique()} dates)")
