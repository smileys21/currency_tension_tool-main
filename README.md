# Currency Tension Engine (CTE)

A daily macro dashboard that places the eight major currencies (USD, EUR, JPY,
GBP, CHF, CAD, AUD, NZD) on a two-axis map:

- **Axis 1 — fundamental trajectory** (deteriorating ↔ improving): growth,
  inflation, external, fiscal pillars.
- **Axis 2 — valuation & policy stretch** (cheap ↔ maxed): policy room, carry,
  REER valuation.

Scoring uses dual-horizon z-scores (a ~10-year structural window and a ~2-year
regime window); the divergence between them is the signal. Carry and positioning
are computed **pairwise** across the 28-cross grid — USD is one leg of eight, not
a hub. See `currency_engine_spec_v4.md` for the full design.

## Status

Build proceeds in stages so real data can be sanity-checked before scoring.

| Stage | State |
|-------|-------|
| 0 · Source reachability | ✅ 13/13 sources reachable |
| 1 · Ingestion adapters | ✅ Complete — FX/commodities, sovereign yields (9), CFTC TFF, BIS REER, FRED backbone, OECD-SDMX + Eurostat gap-fillers all verified against live data. 8/9 cross-country gaps closed; only Japan CPI open (needs a JP-specific source). |
| 2 · Transform (dual-horizon z, pairwise) | ⬜ next |
| 3 · Scoring (axis compositor, weighted sliders) | ⬜ |
| 4 · Flags · 5 · Commentary · 6 · Streamlit · 7 · Scheduler | ⬜ |

## Data sources

| Block | Source | Notes |
|-------|--------|-------|
| FX spot + commodities | Yahoo (`yfinance`) | 7 USD legs + DXY; commodities overlay-only |
| Sovereign yields (2Y/3Y/10Y) | 9 official CB/DMO endpoints | US Treasury, ECB, Bundesbank, MoF JGB, BoE GLC, SNB spot curve, BoC, RBA, RBNZ |
| Positioning | CFTC TFF (Futures-Only) | leveraged-money + asset-manager net, weekly |
| REER | BIS broad real EER | monthly |
| Macro backbone + US real yields | FRED | CLI, GDP, unemployment, core CPI, policy rate, TIPS reals |

## Setup

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_free_key      # https://fred.stlouisfed.org
```

Verify the ingestion layer against live data:

```bash
python scripts/verify_yields.py        # 9 sovereign-yield sources
python -m cte.adapters.cftc_tff        # positioning
python -m cte.adapters.bis_reer        # REER
python scripts/verify_fred.py          # FRED backbone (needs key)
```

Daily refresh runs unattended via `.github/workflows/refresh.yml` (set
`FRED_API_KEY` as a repo secret).

## Known limitations / deferred work

- **Mixed yield-curve basis.** CHF/EUR/GBP legs are fitted zero-coupon *spot*
  rates; USD is the Treasury *par* curve; CAD/AUD/NZD are benchmark-bond YTMs.
  Immaterial for the z-score/relative scoring the engine uses (a few-bp level
  offset is mean-centered out), but a strict apples-to-apples standardization
  onto one basis is a possible future upgrade.
- **Macro backbone assembled from three sources; 8/9 gaps closed.** `config.FRED_SERIES`
  is verified live against actual observations (28/31 fresh; GBP unemployment/policy
  and NZ GDP are the freshest FRED carries but lag ~1–2 periods). FRED's OECD mirrors
  froze for CLI (EUR/CHF/NZD), core CPI (GBP/JPY/CAD/AUD/NZD), and euro-area
  unemployment. These are now closed by `cte.adapters.oecd` (CLI business-confidence
  proxy + headline CPI) and `cte.adapters.eurostat` (EA unemployment).
  `cte.adapters.macro.build_macro_backbone()` unifies FRED + OECD + Eurostat + e-Stat
  into one panel. **All macro inputs are now sourced** — Japan CPI is closed via
  `cte.adapters.estat` (Statistics Bureau e-Stat, headline CPI YoY back to 1971), so
  `config.MACRO_GAPS_REMAINING` is empty.
- **CPI is a core/headline mix by necessity.** A consistent monthly *core* CPI across
  all eight does not exist in free structured data (AUS/NZ publish CPI only quarterly;
  Japan/Canada core cuts aren't carried). Inflation input is therefore FRED **core**
  for USD/EUR/CHF and OECD **headline** YoY for GBP/CAD/AUD/NZD. If you'd prefer a
  uniform definition, OECD headline is available for USD/EUR/CHF too (slightly staler).
- **CLI is a confidence proxy for EUR/CHF/NZD.** OECD discontinued the *composite* CLI
  for these three, keeping only business/consumer confidence; the adapter uses
  amplitude-adjusted business confidence (BCICP) as the leading-indicator proxy.
- **US 2Y real is derived, not sourced.** There is no DFII2 series (shortest TIPS
  constant maturity is 5Y), so US 2Y real = 2Y nominal − core CPI in the transform
  layer, consistent with how the other legs' real short rate is built.
- **BoE full history opt-in.** `fetch_boe` pulls the current-month GLC file;
  wiring the 38 MB full-history archive is deferred until scoring needs the long
  z-score baseline for GBP.
- **CHF has no 3Y node.** The SNB spot-curve source publishes 2Y/5Y/10Y/20Y; 2Y
  (carry) and 10Y (fiscal) are covered, and the 3Y fallback was removed.
