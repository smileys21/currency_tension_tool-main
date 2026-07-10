# Deploying the Currency Tension Engine

The app serves a small **pre-computed snapshot** that is committed to the repo; the
big raw data cache is rebuilt daily by a GitHub Action and never committed. So the
public dashboard does no data pulls and makes no API calls — it just renders the
latest committed snapshot. Cost and load stay flat no matter how many people use it.

```
GitHub Action (daily, your keys)                 Streamlit Community Cloud (public)
  backfill → engine snapshot → commentary   ──►    reads committed snapshot only
  commits snapshot back to repo             ──►    auto-redeploys on each commit
```

## 1. Push to GitHub

```bash
cd currency_tension_engine
git init && git add -A
git commit -m "Currency Tension Engine"
git branch -M main
git remote add origin https://github.com/<you>/currency-tension-engine.git
git push -u origin main
```

The initial commit already includes a working snapshot (`data/cache/tension_map.parquet`,
`pillar_scores`, `overlays`, both carry grids, `warnings.json`, `commentary.md`), so the
app has data the moment it deploys — before the Action ever runs.

## 2. Add repo secrets (for the daily Action)

Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Used for |
|---|---|
| `FRED_API_KEY` | US CPI/unemployment/real-yield, EUR GDP/CPI, CHF CPI |
| `ESTAT_APP_ID` | Japan CPI (e-Stat) |
| `ANTHROPIC_API_KEY` | the once-daily cached commentary |

The Action (`.github/workflows/refresh.yml`) runs 06:30 UTC on weekdays, rebuilds the
snapshot, and commits it back. Trigger it once manually (**Actions → daily-refresh →
Run workflow**) to confirm it's green.

## 3. Deploy on Streamlit Community Cloud

1. https://share.streamlit.io → **New app** → pick the repo/branch.
2. Main file path: `streamlit_app.py`.
3. Deploy. No secrets are required on Streamlit's side — the public app never calls
   an API. (The rebuild button is hidden unless the raw cache and keys are present, so
   it only appears when you run locally.)

Each time the Action commits a fresh snapshot, Streamlit Cloud redeploys automatically,
so the live app tracks the daily refresh with no further action.

## Running locally

```bash
pip install -r requirements.txt
export FRED_API_KEY=…  ESTAT_APP_ID=…            # ANTHROPIC_API_KEY optional
python -m scripts.backfill                        # one-time full-history seed
python -m cte.scoring.engine                       # build the snapshot
streamlit run streamlit_app.py
```
