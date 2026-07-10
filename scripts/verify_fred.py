"""Verify every candidate FRED series against the live API.

Run after exporting your key:  export FRED_API_KEY=xxxxx
Reports, per (currency, metric): series id, obs count, latest date + value, and a
freshness verdict against the metric's expected cadence. Dead / discontinued ids
are listed at the end so they can be substituted in config.FRED_SERIES.
"""
import sys, pathlib, datetime as dt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from cte.adapters import fred
from cte.config import FRED_SERIES, FRED_US_REAL

def _allowed_stale_days(dates: pd.Series) -> tuple[int, str]:
    """Infer cadence from the median gap between observations and allow ~2 periods
    plus a release-lag buffer. Auto-handles daily / monthly / quarterly series so
    quarterly data isn't false-flagged as stale."""
    d = dates.sort_values().tail(24)
    if len(d) < 3:
        return 120, "?"
    gap = d.diff().dt.days.dropna().median()
    if gap <= 4:
        return 10, "daily"
    if gap <= 45:
        return 100, "monthly"       # ~2 months + OECD release lag
    return 200, "quarterly"          # ~1 quarter + release lag

def main() -> None:
    try:
        key = fred.get_api_key()
    except fred.FredKeyMissing as e:
        raise SystemExit(str(e))

    sess = fred.make_session()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    plan = [(c, m, s) for c, ms in FRED_SERIES.items() for m, s in ms.items()]
    plan += [("USD", m, s) for m, s in FRED_US_REAL.items()]

    print(f"{'ccy':<5}{'metric':<10}{'series':<20}{'obs':>6}  "
          f"{'latest':<12}{'value':>12}  verdict")
    print("-" * 78)
    dead = []
    for ccy, metric, sid in plan:
        try:
            d = fred.fetch_series(sid, session=sess, api_key=key)
            if d.empty:
                print(f"{ccy:<5}{metric:<10}{sid:<20}{0:>6}  {'—':<12}{'—':>12}  EMPTY")
                dead.append((ccy, metric, sid, "empty"))
                continue
            last = d.sort_values("date").iloc[-1]
            stale = (today - last["date"].normalize()).days
            limit, cadence = _allowed_stale_days(d["date"])
            verdict = "ok" if stale <= limit else f"STALE {stale}d>{limit}"
            if stale > limit:
                dead.append((ccy, metric, sid, f"stale {stale}d"))
            print(f"{ccy:<5}{metric:<10}{sid:<20}{len(d):>6}  "
                  f"{str(last['date'].date()):<12}{last['value']:>12.3f}  {verdict}")
        except Exception as e:  # noqa: BLE001
            print(f"{ccy:<5}{metric:<10}{sid:<20}{'—':>6}  {'—':<12}{'—':>12}  "
                  f"FAIL {type(e).__name__}")
            dead.append((ccy, metric, sid, type(e).__name__))

    print("-" * 78)
    if dead:
        print(f"\n{len(dead)} series need substitution:")
        for ccy, metric, sid, why in dead:
            print(f"  {ccy}/{metric}  {sid}  — {why}")
    else:
        print("\nAll candidate FRED series live and fresh.")


if __name__ == "__main__":
    main()
