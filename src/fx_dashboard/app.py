"""Streamlit entry point. Run with: streamlit run src/fx_dashboard/app.py"""

from datetime import date

import streamlit as st

from fx_dashboard import theme  # noqa: F401 - registers the Plotly template
from fx_dashboard.countries import COUNTRIES
from fx_dashboard.data import load_all, load_country
from fx_dashboard.series import DOLLAR_INDEX, FED_TARGET_IDS, FX_PAIRS, MACRO_VARS
from fx_dashboard.views import dollar_tab, fed_tab, macro_tab, pair_regimes_tab, pairs_tab

TITLE = "Nyasha Mugabe FX Macro Dashboard"

st.set_page_config(page_title=TITLE, layout="wide")


@st.cache_data(ttl=60 * 60 * 12, show_spinner="Downloading data from FRED and Yahoo Finance...")
def get_data(series_ids: tuple[str, ...]) -> dict:
    # series_ids is only a cache key: adding a series to the catalogue forces a fresh download.
    return load_all()


@st.cache_data(ttl=60 * 60 * 12, show_spinner="Downloading this economy's data...")
def get_country(code: str, sources: str) -> dict:
    # sources is only a cache key: editing a country's catalogue forces a fresh download.
    return load_country(COUNTRIES[code])


def country_data(code: str) -> dict:
    return get_country(code, repr(COUNTRIES[code].sources))


data = get_data(tuple(MACRO_VARS) + tuple(FX_PAIRS) + (DOLLAR_INDEX,) + FED_TARGET_IDS)

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
tab_macro, tab_dollar, tab_pairs, tab_pair_regimes, tab_fed = st.tabs(
    [
        "Macro Variables",
        "Dollar Index Regimes",
        "FX Pairs & Scenarios",
        "Pair Regimes",
        "Fed Hiking vs Cutting",
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
