"""Streamlit entry point. Run with: streamlit run src/fx_dashboard/app.py"""

import os
from datetime import date

import streamlit as st

from fx_dashboard import theme  # noqa: F401 - registers the Plotly template
from fx_dashboard.countries import COUNTRIES
from fx_dashboard.data import load_all, load_country
from fx_dashboard.series import (
    DOLLAR_INDEX,
    FED_TARGET_IDS,
    FX_PAIRS,
    G10_PAIRS,
    MACRO_VARS,
    US_EXTRA,
)
from fx_dashboard.views import (
    dollar_tab,
    fed_tab,
    g10_tab,
    macro_tab,
    pair_regimes_tab,
    pairs_tab,
    regression_tab,
)

TITLE = "Nyasha Mugabe FX Macro Dashboard"

st.set_page_config(page_title=TITLE, layout="wide")

# A FRED API key stored in Streamlit secrets (Settings -> Secrets on Community Cloud) is
# passed to the data loader through the environment. Without one the keyless CSV is used.
try:
    if "FRED_API_KEY" in st.secrets:
        os.environ.setdefault("FRED_API_KEY", str(st.secrets["FRED_API_KEY"]))
except Exception:  # noqa: BLE001 - no secrets file locally is fine
    pass


@st.cache_data(ttl=60 * 60 * 12, show_spinner="Downloading data from FRED and Yahoo Finance...")
def get_data(series_ids: tuple[str, ...], has_key: bool) -> dict:
    # Both arguments are only cache keys: adding a series to the catalogue, or adding a FRED
    # API key, forces a fresh download instead of reusing a cached failure.
    return load_all()


@st.cache_data(ttl=60 * 60 * 12, show_spinner="Downloading this economy's data...")
def get_country(code: str, sources: str, has_key: bool) -> dict:
    # sources is only a cache key: editing a country's catalogue forces a fresh download.
    return load_country(COUNTRIES[code])


HAS_FRED_KEY = bool(os.environ.get("FRED_API_KEY", "").strip())


def country_data(code: str) -> dict:
    return get_country(code, repr(COUNTRIES[code].sources), HAS_FRED_KEY)


data = get_data(
    tuple(MACRO_VARS)
    + tuple(FX_PAIRS)
    + tuple(G10_PAIRS)
    + US_EXTRA
    + (DOLLAR_INDEX,)
    + FED_TARGET_IDS,
    HAS_FRED_KEY,
)

with st.sidebar:
    st.header("Settings")
    start = st.date_input(
        "Analysis start",
        value=date(2006, 1, 1),
        min_value=date(1990, 1, 1),
        max_value=date.today(),
        help="Applies to every chart and model. The broad dollar index starts in Jan 2006.",
    )
    if st.button("Refresh data", width="stretch"):
        get_data.clear()
        get_country.clear()
        st.rerun()
    if data["errors"]:
        st.warning(
            "Some series failed to load:\n\n"
            + "\n".join(f"- **{k}**: {v}" for k, v in data["errors"].items())
        )
    st.caption(
        "Data: FRED (St. Louis Fed); Yahoo Finance for USDCLP; BIS and OECD where a FRED "
        "series for another G10 economy has stopped updating. Cached for 12 hours."
    )
    st.caption(
        "FRED API key: **found**, using the FRED API."
        if HAS_FRED_KEY
        else "FRED API key: **not set**, using FRED's keyless download."
    )

missing = [k for k in MACRO_VARS if k not in data["macro"]]
if missing or data["dollar"] is None:
    st.error(
        "Required series failed to load: "
        + ", ".join(missing + ([DOLLAR_INDEX] if data["dollar"] is None else []))
        + ". Press Refresh data to retry."
    )
    st.stop()

st.title(TITLE)
fed_target = data.get("fed_target")
tab_macro, tab_dollar, tab_pairs, tab_pair_regimes, tab_fed, tab_g10, tab_reg = st.tabs(
    [
        "Macro Variables",
        "Dollar Index Regimes",
        "FX Pairs & Scenarios",
        "Pair Regimes",
        "Fed Hiking vs Cutting",
        "G10 vs US",
        "Regression",
    ]
)
with tab_macro:
    macro_tab.render(data["macro"], start, country_data)
with tab_dollar:
    dollar_tab.render(data["dollar"], data["macro"], start, fed_target)
with tab_pairs:
    pairs_tab.render(data["pairs"], data["dollar"], data["macro"], start)
with tab_pair_regimes:
    pair_regimes_tab.render(data["pairs"], data["dollar"], data["macro"], start, fed_target)
with tab_fed:
    fed_tab.render(data["pairs"], data["dollar"], data["macro"], start, fed_target)
with tab_g10:
    g10_tab.render(
        data["macro"],
        data["pairs"] | data.get("g10_pairs", {}),
        data.get("us_extra", {}),
        start,
        fed_target,
        country_data,
    )
with tab_reg:
    regression_tab.render(
        data["macro"],
        data["pairs"] | data.get("g10_pairs", {}),
        data.get("us_extra", {}),
        start,
        fed_target,
        country_data,
    )
