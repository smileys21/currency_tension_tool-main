"""Seed the tension-map history (one-time monthly backfill).

Usage:
  python -m scripts.backfill_history               # always (re)build
  python -m scripts.backfill_history --if-missing  # no-op when the committed
                                                   # snapshot_history.parquet exists
                                                   # (the daily engine appends to it)
Run AFTER scripts.backfill (needs the raw feature cache).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cte.adapters.base import CACHE_DIR                      # noqa: E402
from cte.scoring.history import HISTORY_NAME, backfill      # noqa: E402

if __name__ == "__main__":
    target = CACHE_DIR / f"{HISTORY_NAME}.parquet"
    if "--if-missing" in sys.argv and target.exists():
        print(f"history present ({target.name}) — skipping backfill; "
              "the engine appends daily.")
        sys.exit(0)
    h = backfill()
    print(f"backfilled {h.date.nunique()} month-ends, "
          f"{h.ccy.nunique()} currencies, {len(h)} rows "
          f"({h.date.min().date()} -> {h.date.max().date()})")
