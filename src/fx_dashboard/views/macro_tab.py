"""Tab 1: toggleable macro variables for the US and the other G10 economies."""

from collections.abc import Callable

import pandas as pd
import streamlit as st

from fx_dashboard.analysis import pct
from fx_dashboard.countries import COUNTRIES
from fx_dashboard.series import MACRO_VARS, MacroVar
from fx_dashboard.theme import DRIVER_COLORS, line, show

VIEWS = {
    "Level": "level",
    "Change vs previous period": "change",
    "Year-on-year change": "yoy",
}
US = "US"
ECONOMIES = {US: "United States"} | {code: c.name for code, c in COUNTRIES.items()}


def display_series(var: MacroVar, s: pd.Series, view: str) -> tuple[pd.Series, str]:
    """Resample to display frequency and apply the chosen transform. Returns (series, unit)."""
    s = s.resample("QE" if var.freq == "Q" else "ME").last().dropna()
    if view == "level":
        return s, var.unit
    periods = 1 if view == "change" else (4 if var.freq == "Q" else 12)
    if var.kind == "rate":
        return s.diff(periods), "pp"
    return pct(s, periods), "%"


def _fmt_level(var: MacroVar, value: float, code: str) -> str:
    if var.kind == "rate":
        return f"{value:.2f}%"
    if var.key == "GDP":
        return f"${value:,.0f}bn" if code == US else f"{value:,.0f}"
    return f"{value:,.1f}"


def _us_data(macro: dict[str, pd.Series]) -> dict:
    return {
        "macro": macro,
        "sources": {k: f"FRED {v.fred_id}" for k, v in MACRO_VARS.items()},
        "notes": {},
        "errors": {},
    }


def render(macro: dict[str, pd.Series], start, load_country: Callable[[str], dict]) -> None:
    code = st.segmented_control(
        "Economy",
        list(ECONOMIES),
        format_func=ECONOMIES.get,
        default=US,
        key="macro_country",
    )
    code = code or US  # clicking the selected option clears it; fall back to the US
    if code == US:
        variables, data = MACRO_VARS, _us_data(macro)
    else:
        variables, data = COUNTRIES[code].vars, load_country(code)

    st.subheader(f"{ECONOMIES[code]} macro variables")
    if code == US:
        st.caption(
            "Switch each variable on or off. Source: FRED. The 2Y yield, spread, CPI and fed "
            "funds are shown monthly (month-end / monthly average); real GDP is quarterly."
        )
    else:
        if variables["DGS2"].short == "3M rate":
            short_end = (
                "FRED has no 2Y government yield for this economy, so the **3-month interbank "
                "rate** stands in for the 2Y yield and the curve is **10Y yield minus the 3M "
                "rate**. "
            )
        else:
            short_end = (
                "The 2Y yield and 10Y–2Y spread come from the Bank of England's gilt curve "
                "(month-end spot yields), since FRED does not carry them. "
            )
        st.caption(
            "Switch each variable on or off. FRED first; another official source only where "
            f"FRED lacks the series or has stopped updating it. {short_end}CPI is not "
            "seasonally adjusted: use year-on-year change."
        )
    for key, err in data["errors"].items():
        st.warning(f"{variables[key].label} failed to load: {err}")

    available = [k for k in variables if k in data["macro"]]
    cols = st.columns(len(variables))
    enabled = [
        key
        for col, (key, var) in zip(cols, variables.items(), strict=True)
        if col.toggle(var.label, value=True, key=f"macro_on_{key}", disabled=key not in available)
        and key in available
    ]
    view_label = st.radio("Show as", list(VIEWS), horizontal=True, key="macro_view")
    view = VIEWS[view_label]

    if not enabled:
        st.info("All variables are switched off. Toggle one on above.")
        return

    for col, key in zip(st.columns(len(enabled)), enabled, strict=True):
        var, s = variables[key], data["macro"][key]
        yoy, unit = display_series(var, s, "yoy")
        yoy = yoy.dropna()
        col.metric(
            var.label,
            _fmt_level(var, s.iloc[-1], code),
            delta=f"{yoy.iloc[-1]:+.2f}{unit} y/y" if not yoy.empty else None,
            delta_color="off",
            delta_arrow="off",
            help=f"Latest observation: {s.index[-1]:%d %b %Y}. Source: {data['sources'][key]}",
        )

    frames = {}
    grid = st.columns(2)
    for i, key in enumerate(enabled):
        var = variables[key]
        s, unit = display_series(var, data["macro"][key], view)
        s = s[s.index >= pd.Timestamp(start)].dropna()
        frames[f"{var.label} ({unit})"] = s
        with grid[i % 2]:
            st.markdown(f"**{var.label}** · {view_label.lower()}")
            show(
                line(
                    s, DRIVER_COLORS[key], y_title=unit, name=var.label, zero_line=view != "level"
                ),
                height=300,
            )

    with st.expander("Sources"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Variable": variables[k].label,
                        "Source": data["sources"][k],
                        "From": f"{data['macro'][k].index.min():%b %Y}",
                        "Latest": f"{data['macro'][k].index.max():%b %Y}",
                        "Why not FRED": data["notes"].get(k, ""),
                    }
                    for k in available
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Data table"):
        table = pd.concat(frames, axis=1).sort_index(ascending=False)
        st.dataframe(table, width="stretch")
        st.download_button(
            "Download CSV",
            table.to_csv().encode(),
            file_name=f"macro_{code.lower()}_{view}.csv",
            mime="text/csv",
        )
