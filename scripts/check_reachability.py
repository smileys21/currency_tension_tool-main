"""
Reachability probe for the Currency Tension Engine ingestion layer.

Hits the *actual resource path* for each source (not just a host ping), so a
PASS means the real data route is live, not merely that DNS resolves. FRED is
probed without a key — a 400 "missing api_key" still proves the host/route is
up; the keyed pull is verified in the FRED adapter stage.

Run:  python scripts/check_reachability.py
"""
from __future__ import annotations

import sys
import time
import datetime as dt
import requests

UA = {"User-Agent": "Mozilla/5.0 (CTE reachability probe; contact: research desk)"}
TIMEOUT = 45
CUR_YEAR = dt.date.today().year

# (label, ccy/role, method, url, ok_predicate)
# ok_predicate(resp) -> (bool ok, str note)
def _ok_200(r):
    return (r.status_code == 200, f"{len(r.content)} bytes")

def _ok_fred(r):
    # 400 with api_key message = host live, key simply absent
    if r.status_code == 400 and (b"api_key" in r.content or b"Bad Request" in r.content):
        return (True, "host live (api_key required, as expected)")
    return (r.status_code == 200, f"unexpected status {r.status_code}")

def _ok_json_nonempty(r):
    if r.status_code != 200:
        return (False, f"status {r.status_code}")
    try:
        j = r.json()
        return (True, f"json ok ({len(str(j))} chars)")
    except Exception as e:
        return (False, f"not json: {e}")

PROBES = [
    # ---- macro backbone ----
    ("FRED API",            "macro", "GET",
     "https://api.stlouisfed.org/fred/series?series_id=GDP&file_type=json",
     _ok_fred),
    ("Yahoo finance query", "spot/cmdty", "GET",
     "https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?range=5d&interval=1d",
     _ok_json_nonempty),

    # ---- eight currencies / nine sovereign-yield endpoints (EUR has two legs) ----
    ("US Treasury par curve (CSV)", "USD", "GET",
     f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{CUR_YEAR}/all?type=daily_treasury_yield_curve&field_tdr_date_value={CUR_YEAR}&page&_format=csv",
     _ok_200),
    ("Bank of Canada Valet", "CAD", "GET",
     "https://www.bankofcanada.ca/valet/observations/BD.CDN.2YR.DQ.YLD/json?recent=5",
     _ok_json_nonempty),
    ("ECB Data Portal SDMX (YC)", "EUR-area", "GET",
     "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y?lastNObservations=5&format=jsondata",
     _ok_json_nonempty),
    ("Bundesbank time-series REST", "EUR-Bund", "GET",
     "https://api.statistiken.bundesbank.de/rest/data/BBSIS/D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A?lastNObservations=5&format=json",
     _ok_json_nonempty),
    ("BoE IADB (gilt yield)", "GBP", "GET",
     "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2026&Dateto=now&SeriesCodes=IUDMNZC&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N",
     _ok_200),
    ("Japan MoF JGB CSV (jgbcm_all)", "JPY", "GET",
     "https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv",
     _ok_200),
    ("RBA F2 capital-mkt yields", "AUD", "GET",
     "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv",
     _ok_200),
    ("RBNZ B2 wholesale rates", "NZD", "GET",
     "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b2/hb2-daily.csv",
     _ok_200),
    ("SNB data portal API", "CHF", "GET",
     "https://data.snb.ch/api/cube/rendoblim/data/csv/en",
     _ok_200),

    # ---- positioning + valuation ----
    ("CFTC TFF (Socrata)", "positioning", "GET",
     "https://publicreporting.cftc.gov/resource/gpe5-46if.json?$limit=1",
     _ok_json_nonempty),
    ("BIS stats REER (v1)", "valuation", "GET",
     "https://stats.bis.org/api/v1/data/BIS,WS_EER,1.0/M.R.B.US?lastNObservations=5&detail=dataonly",
     _ok_200),
]

def main():
    print(f"CTE reachability probe — {dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"year={CUR_YEAR}  timeout={TIMEOUT}s\n")
    results = []
    for label, role, method, url, pred in PROBES:
        t0 = time.time()
        try:
            r = requests.request(method, url, headers=UA, timeout=TIMEOUT)
            ok, note = pred(r)
            status = r.status_code
        except Exception as e:
            ok, note, status = False, f"{type(e).__name__}: {e}", "ERR"
        dt_ms = int((time.time() - t0) * 1000)
        mark = "PASS" if ok else "FAIL"
        results.append((mark, label, role, status, dt_ms, note, url))
        print(f"[{mark}] {label:<32} {str(status):>4}  {dt_ms:>5}ms  {role:<12} {note}")

    npass = sum(1 for r in results if r[0] == "PASS")
    print(f"\n{npass}/{len(results)} endpoints reachable")
    fails = [r for r in results if r[0] == "FAIL"]
    if fails:
        print("\nFAILURES (need an alternate route or header tweak):")
        for _, label, role, status, _, note, url in fails:
            print(f"  - {label} [{role}] status={status} :: {note}\n      {url}")
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
