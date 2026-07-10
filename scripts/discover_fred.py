"""Find live FRED series for a metric/country by querying the search API.

Ranks candidates by last-updated (freshest first) so we replace dead OECD mirrors
with series that are actually maintained. Usage:
    python scripts/discover_fred.py "core cpi united kingdom" Monthly
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cte.adapters import fred
from cte.config import FRED_BASE


def search(text: str, frequency: str | None = None, limit: int = 8):
    sess = fred.make_session()
    key = fred.get_api_key()
    params = {
        "search_text": text, "api_key": key, "file_type": "json",
        "order_by": "last_updated", "sort_order": "desc", "limit": 40,
    }
    r = sess.get(FRED_BASE + "/series/search", params=params, timeout=45)
    r.raise_for_status()
    out = []
    for s in r.json().get("seriess", []):
        if frequency and s.get("frequency_short") != frequency:
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    text = sys.argv[1]
    freq = sys.argv[2] if len(sys.argv) > 2 else None
    for s in search(text, freq):
        print(f"{s['id']:<22}{s.get('frequency_short',''):<4}"
              f"{s['last_updated'][:10]:<12}{s['title'][:70]}")
