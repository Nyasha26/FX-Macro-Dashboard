# fx-dashboard

Streamlit dashboard analysing how US macro variables relate to FX movements.

## Tabs

1. **Macro Variables** – pick an economy (US, Australia, Canada, France, Germany, Italy,
   Japan, New Zealand, Norway, Sweden, Switzerland, United Kingdom) and toggle its policy rate, short rate,
   yield-curve spread, CPI and real GDP on or off; view as level, period change or
   year-on-year. For the US these are DGS2, T10Y2Y, CPIAUCSL, FEDFUNDS and GDPC1.
   FRED has no 2Y government yield outside the US, so most economies show the 3-month
   interbank rate and the 10Y–3M spread instead; the UK uses the Bank of England's gilt
   curve for a true 2Y yield and 10Y–2Y spread. Each variable's source is listed in the
   tab (see `src/fx_dashboard/countries.py`).
2. **Dollar Index Regimes** – how the Fed's nominal broad dollar index (DTWEXBGS) behaves
   when each variable is going up, flat or down: average move by regime, hit rates,
   a two-variable regime matrix and a regime-shaded timeline. Quarterly, because GDP is.
3. **FX Pairs & Scenarios** – toggle USDEUR, USDJPY, USDCHF, USDGBP, USDZAR, USDBRL,
   USDCNH, USDMXN and USDCLP. Shows a snapshot table, rebased performance and a
   macro-shock scenario: step fed funds and the 2Y yield in 2bp increments, GDP in
   1% increments and CPI in 0.1pp increments, and see the implied move in each pair
   from a quarterly regression on those variables (plus the 10Y–2Y spread).
4. **Pair Regimes** – every pair (and the broad dollar index) across each variable's
   regimes: heatmap, regime winners and losers, summary table.
5. **Fed Hiking vs Cutting** – USD returns against each currency and the broad dollar
   index while the Fed is hiking, on hold or cutting: averages, annualised returns,
   hit rates, a hiking-minus-cutting t-stat and cycle-by-cycle cumulative moves.

Fed regimes are dated from the Fed's policy target (FRED DFEDTAR to Dec 2008, then the
upper bound DFEDTARU), not the monthly-average effective rate. By default a pause between
two hikes (or two cuts) of up to 4 quarters counts as part of the cycle.

All pairs are quoted as foreign currency per 1 USD, so up = stronger dollar.

## Data

- FRED public CSV endpoints (no API key) for macro data, the dollar index and most pairs.
- Yahoo Finance for USDCLP (FRED does not publish CLP).
- Bank of England month-end gilt curve for the UK 2Y yield and 10Y–2Y spread (not on FRED).
- Policy rates are the official central-bank rates: FRED's ECB deposit rate for the euro
  area, the Bank of England database for Bank Rate, and the BIS for the others (FRED
  only has monthly-average interbank rates for these, which lag rate changes).
- BIS (CPI) and OECD (New Zealand GDP) only where the FRED series for
  another economy has stopped updating; the loader checks freshness and falls back automatically.
- **USDCNH uses onshore USD/CNY (FRED DEXCHUS) as a proxy**: no free source provides
  offshore CNH history.

Data is cached for 12 hours; use **Refresh data** in the sidebar to reload.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run

```powershell
streamlit run src/fx_dashboard/app.py
```

## Develop

```powershell
pytest
ruff check .
```

## Layout

```
src/fx_dashboard/
  app.py          Streamlit entry point: sidebar, data load, tabs
  series.py       macro variables and FX pair definitions
  data.py         FRED / Yahoo loaders
  analysis.py     quarterly drivers, regimes, sensitivity regressions, scenarios
  theme.py        Plotly template and fixed colour assignments
  views/          one module per tab
tests/            unit tests for analysis code
```

## Publish online (Streamlit Community Cloud, free)

1. Push this folder to a GitHub repository (private is fine; the app can still be public).
2. Sign in at <https://share.streamlit.io> with GitHub and click **Create app**.
3. Choose the repository and branch `main`, set **Main file path** to
   `src/fx_dashboard/app.py`, and under **Advanced settings** pick Python 3.12.
4. Click **Deploy**. Streamlit installs `requirements.txt` (this project plus its
   dependencies) and gives the app a public `*.streamlit.app` URL.
5. In the app's **Settings → Sharing**, make sure it is set to **Public**.

Every push to `main` redeploys automatically. Free apps go to sleep after a period with
no visitors; the next visitor wakes them with one click (about 30 seconds).

