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
6. **G10 vs US** – pick any of the 11 other G10 economies and a variable (policy rate,
   short rate, yield curve, inflation, growth). Each side gets its own regime (e.g. Fed
   hiking vs BoJ on hold); a 3×3 grid shows USD/that currency in every combination, plus
   the spot when the US-minus-foreign gap rises or falls and a gap-vs-spot regression.
   Like for like: 3M-rate economies are compared with the US 3M bill (FRED DTB3) and
   10Y−3M spread (T10Y3M); the UK with the US 2Y and 10Y−2Y. Adds USDAUD, USDCAD, USDNZD,
   USDNOK and USDSEK from FRED.
   **2Y govt yield** (Germany, Japan, UK): daily US 2Y Treasury minus the 2Y Bund / JGB /
   gilt, with the spot, a 52-week rolling correlation, spot % per 10bp of differential
   (1Y, 3Y, whole sample), the spot level implied by the gap, and a daily CSV.
7. **Regression** – what drives USD against each G10 currency. Quarterly % change in the
   pair regressed (OLS, robust HC1 errors) on the quarterly change in inflation (CPI y/y),
   growth (real GDP y/y) and the 2Y yield (3M rate where no 2Y is available), toggled
   between the **US-minus-foreign gap** and **US and foreign variables separately**, same or
   next quarter. Shows R², coefficients per 1pp and per typical move with t-stats, a test of
   whether only the gap matters (US coefficient = minus the foreign one), each driver's
   contribution to every quarter's move, rolling coefficients, and the same model across all
   11 economies.

### 2Y government yields (bundled)

FRED has no 2Y government yields outside the US, so they come from downloaded files:
Bundesbank (current 2Y Federal note), Ministry of Finance Japan (JGB 2Y) and the Bank of
England daily nominal spot curve (2.0 years). Only the 2Y columns are kept, in
`src/fx_dashboard/resources/yields_2y.csv`. To update, download fresh copies of
`2yr Bund Yields.csv`, `JGB_YIELDS .csv` and `Gilt spot curve.csv` into one folder, then:

```powershell
python -m fx_dashboard.yields2y "C:\path\to\folder"
git add src/fx_dashboard/resources/yields_2y.csv; git commit -m "Update 2Y yields"; git push
```

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
6. **FRED API key (needed on Streamlit Cloud).** FRED's keyless CSV download times out from
   cloud servers. Get a free key at <https://fredaccount.stlouisfed.org/apikeys>, then in
   the app's **Settings → Secrets** paste `FRED_API_KEY = "your-key"` and save. Locally the
   key is optional (set the `FRED_API_KEY` environment variable to use it).

Every push to `main` redeploys automatically. Free apps go to sleep after a period with
no visitors; the next visitor wakes them with one click (about 30 seconds).

