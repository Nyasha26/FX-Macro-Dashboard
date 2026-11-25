"""Tab 7: regression of USD spot moves on US and foreign inflation, growth and 2Y yields.

Two specifications: the US-minus-foreign gap in each variable, or the US and foreign
variables as separate regressors. Quarterly, because GDP is.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fx_dashboard.analysis import (
    Regression,
    equal_and_opposite_pvalue,
    ols,
    quarterly_returns,
    rolling_ols,
)
from fx_dashboard.countries import COUNTRIES, Country
from fx_dashboard.theme import AXIS, DIVERGING, DRIVER_COLORS, INK, MUTED, NEUTRAL, SERIES, show
from fx_dashboard.views.g10_tab import Y2, quarterly_values, sides

DRIVER_NAMES = {"RATE": "2Y yield", "CPI": "Inflation", "GDP": "Growth"}
DRIVER_HELP = {
    "RATE": "2Y government yield (3-month rate where no 2Y yield is available)",
    "CPI": "CPI inflation, y/y %",
    "GDP": "Real GDP growth, y/y %",
}
COLORS = {"RATE": DRIVER_COLORS["DGS2"], "CPI": DRIVER_COLORS["CPI"], "GDP": DRIVER_COLORS["GDP"]}
SPECS = {"gap": "Gap (US − foreign)", "separate": "US and foreign separately"}
FOREIGN_PATTERN = "/"  # texture marks the foreign side, so identity is never colour alone


@dataclass(frozen=True)
class Driver:
    key: str  # RATE | CPI | GDP
    name: str  # e.g. "2Y yield", "3M rate"
    us: pd.Series  # quarterly: rate level, or y/y % for CPI and GDP
    fx: pd.Series
    us_label: str
    fx_label: str


@dataclass(frozen=True)
class Column:
    """One regressor: which driver, which side, and how to show it."""

    driver: str
    side: str  # gap | us | fx
    label: str  # specific, e.g. "US − Japan 2Y yield gap"
    generic: str  # same across economies, e.g. "2Y yield gap"


def load_drivers(macro, fed_target, us_extra, country: Country, fdata) -> tuple[dict, list, list]:
    """Quarterly US and foreign series for each driver, plus notes and load failures."""
    rate_key = "Y2" if country.code in Y2 else "DGS2"
    out, notes, errors = {}, [], []
    for dk, gk in (("RATE", rate_key), ("CPI", "CPI"), ("GDP", "GDP")):
        us, fx, note = sides(gk, macro, fed_target, us_extra, country, fdata)
        if note:
            notes.append(note)
        if fx is None:
            errors.append(f"{DRIVER_NAMES[dk]}: {fdata['errors'].get(gk, 'not available')}")
            continue
        name = DRIVER_NAMES[dk] if dk != "RATE" or rate_key == "Y2" else "3M rate"
        out[dk] = Driver(
            dk, name, quarterly_values(gk, us), quarterly_values(gk, fx), us.label, fx.label
        )
    return out, notes, errors


def design(drivers: dict, chosen: list[str], spec: str, country: Country):
    """Quarterly changes of the chosen drivers as regressors, with display metadata."""
    X, meta = {}, {}
    for dk in chosen:
        d = drivers.get(dk)
        if d is None:
            continue
        if spec == "gap":
            col = f"{dk}_gap"
            X[col] = (d.us - d.fx).diff()  # NaN across a missing quarter, never a 2-qtr change
            meta[col] = Column(
                dk, "gap", f"{d.name} gap (US − {country.name})", f"{DRIVER_NAMES[dk]} gap"
            )
        else:
            low = d.name if dk == "RATE" else d.name.lower()
            X[f"{dk}_us"] = d.us.diff()
            X[f"{dk}_fx"] = d.fx.diff()
            meta[f"{dk}_us"] = Column(dk, "us", f"US {low}", f"US {DRIVER_NAMES[dk].lower()}")
            meta[f"{dk}_fx"] = Column(
                dk, "fx", f"{country.name} {low}", f"Foreign {DRIVER_NAMES[dk].lower()}"
            )
    return pd.DataFrame(X), meta


def response(px: pd.Series, lead: bool) -> pd.Series:
    ret = quarterly_returns(px)
    return ret.shift(-1) if lead else ret


def fit(y, X, start_ts) -> Regression | None:
    if X.empty:
        return None
    return ols(y[y.index >= start_ts], X[X.index >= start_ts], min_obs=16)


def render(
    macro: dict[str, pd.Series],
    pairs: dict[str, pd.Series],
    us_extra: dict[str, pd.Series],
    start,
    fed_target: pd.Series | None,
    load_country: Callable[[str], dict],
) -> None:
    st.subheader("Regression: what drives USD against each currency")
    st.caption(
        "Quarterly % change in USD/that currency regressed on the quarterly change in "
        "inflation, growth and the 2Y yield, either as the US-minus-foreign gap or with the US "
        "and foreign variables as separate regressors. Pairs are foreign currency per USD: "
        f"positive = USD stronger. From {pd.Timestamp(start):%b %Y} (sidebar)."
    )
    code = (
        st.segmented_control(
            "Economy",
            list(COUNTRIES),
            format_func=lambda c: COUNTRIES[c].name,
            default="JP",
            key="reg_country",
        )
        or "JP"
    )
    c_spec, c_drv, c_resp = st.columns([2, 3, 2])
    spec = (
        c_spec.segmented_control(
            "Regressors", list(SPECS), format_func=SPECS.get, default="gap", key="reg_spec"
        )
        or "gap"
    )
    chosen = c_drv.pills(
        "Drivers",
        list(DRIVER_NAMES),
        selection_mode="multi",
        default=list(DRIVER_NAMES),
        format_func=lambda k: {"RATE": "2Y yield", "CPI": "Inflation (CPI)", "GDP": "Growth (GDP)"}[
            k
        ],
        key="reg_drivers",
        help="; ".join(DRIVER_HELP.values()),
    )
    lead = (
        c_resp.radio(
            "Spot move measured",
            ["Same quarter", "Next quarter"],
            horizontal=True,
            key="reg_response",
            help="'Next quarter' regresses next quarter's spot move on this quarter's macro "
            "changes: does the macro data lead the currency?",
        )
        == "Next quarter"
    )
    if not chosen:
        st.info("Pick at least one driver.")
        return

    country = COUNTRIES[code]
    pair = f"USD{country.currency}"
    px = pairs.get(pair)
    if px is None:
        st.warning(f"{pair} failed to load. Press Refresh data to retry.")
        return
    drivers, notes, errors = load_drivers(macro, fed_target, us_extra, country, load_country(code))
    for msg in errors:
        st.warning(f"{country.name} {msg}. That driver is left out.")
    if "RATE" in drivers and drivers["RATE"].name == "3M rate" and "RATE" in chosen:
        st.info(
            f"No 2Y government yield for {country.name} in the bundled files (they cover "
            "Germany, Japan and the UK), so the rate driver is its 3-month rate against the US "
            "3-month bill."
        )
    for n in notes:
        st.info(n)

    start_ts = pd.Timestamp(start)
    y = response(px, lead)
    X, meta = design(drivers, chosen, spec, country)
    reg = fit(y, X, start_ts)
    if reg is None:
        st.warning("Too few quarters with every chosen driver available to fit the model.")
        return
    meta = {c: meta[c] for c in reg.coef.index}

    _headline(reg, meta, pair)
    _compare_specs(drivers, chosen, y, start_ts, country, pair, spec, reg)
    left, right = st.columns([3, 2])
    with left:
        _coef_chart(reg, meta, pair)
    with right:
        _coef_table(reg, meta, drivers, pair)
    _verdict(reg, meta, pair, lead)
    _contributions(reg, meta, pair)
    _actual_vs_fitted(reg, pair)
    _rolling(reg, meta, pair)
    _all_g10(macro, pairs, us_extra, fed_target, load_country, chosen, spec, lead, start_ts)
    _data(reg, meta, pair, code)
    st.caption(
        "Method: OLS with heteroskedasticity-robust (HC1) standard errors. Regressors are "
        "quarterly changes, in percentage points: the change in the 2Y yield (or 3M rate) "
        "level, and the change in the y/y inflation and growth rates (y/y, so seasonal "
        "patterns in unadjusted foreign CPI cancel). 'Typical move' = one standard deviation of "
        "that driver's quarterly change, so drivers in different units can be ranked. "
        "|t| ≥ 2 is roughly significant at 5%. GDP and CPI are as currently published "
        "(revised), not as first reported, so this describes history rather than a trading "
        "rule. France, Germany and Italy all use USDEUR."
    )


def _p(p: float) -> str:
    return "n/a" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def _headline(reg: Regression, meta: dict, pair: str) -> None:
    cols = st.columns(5)
    first, last = reg.y.index[0].to_period("Q"), reg.y.index[-1].to_period("Q")
    cols[0].metric("Quarters", reg.nobs, help=f"{first} to {last}")
    cols[1].metric(
        "R²",
        f"{reg.r2:.2f}",
        help=f"Share of the quarter-to-quarter variation in {pair} the drivers explain.",
    )
    cols[2].metric(
        "Adjusted R²",
        f"{reg.adj_r2:.2f}",
        help="R² penalised for the number of regressors; use it to compare specifications.",
    )
    cols[3].metric(
        "Model p-value",
        _p(reg.f_pvalue),
        help="Robust joint test that every driver's coefficient is zero. Below 0.05: the "
        "drivers together explain more than chance would.",
    )
    top = reg.coef["t"].abs().idxmax()
    r = reg.coef.loc[top]
    cols[4].metric(
        "Strongest driver",
        meta[top].label,
        delta=f"{r['impact']:+.2f}% per typical move · t {r['t']:.1f}",
        delta_color="off",
        delta_arrow="off",
        help="The driver with the largest |t-stat|.",
    )


def _compare_specs(drivers, chosen, y, start_ts, country, pair, spec, reg) -> None:
    """Fit the other specification on the same quarters and test the gap restriction."""
    other = "separate" if spec == "gap" else "gap"
    X_o, _ = design(drivers, chosen, other, country)
    X_o = X_o.loc[X_o.index.isin(reg.y.index)]
    reg_o = fit(y.loc[reg.y.index], X_o, start_ts)
    sep = reg if spec == "separate" else reg_o
    p = (
        equal_and_opposite_pvalue(sep, [(f"{d}_us", f"{d}_fx") for d in chosen])
        if sep is not None
        else np.nan
    )
    rows = [{"Specification": SPECS[spec] + " (shown)", "R²": reg.r2, "Adjusted R²": reg.adj_r2}]
    if reg_o is not None:
        rows.append({"Specification": SPECS[other], "R²": reg_o.r2, "Adjusted R²": reg_o.adj_r2})
    with st.container(border=True):
        a, b = st.columns([2, 3])
        a.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "R²": st.column_config.NumberColumn(format="%.2f"),
                "Adjusted R²": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        if not np.isfinite(p):
            b.caption("The gap-vs-separate test needs both specifications to fit.")
        elif p >= 0.05:
            b.markdown(
                f"**Gap or separate?** The US and {country.name} coefficients are not "
                f"significantly different from equal-and-opposite (p = {_p(p)}). The data are "
                f"consistent with only the **gap** mattering for {pair}, so the simpler gap "
                "model is the one to read."
            )
        else:
            b.markdown(
                f"**Gap or separate?** The US and {country.name} coefficients are significantly "
                f"different from equal-and-opposite (p = {_p(p)}): the two sides do not move "
                f"{pair} symmetrically, so the **separate** model carries information the gap "
                "hides."
            )
        b.caption(
            "Robust Wald test, on the same quarters, that each US coefficient equals minus "
            "its foreign counterpart, which is what the gap model assumes."
        )


def _bar_style(col: Column) -> dict:
    return dict(
        color=COLORS[col.driver],
        pattern=dict(shape=FOREIGN_PATTERN if col.side == "fx" else "", fgcolor=INK, size=6),
    )


def _coef_chart(reg: Regression, meta: dict, pair: str) -> None:
    st.markdown(f"**{pair} move for a typical quarterly move in each driver** · 95% interval")
    c = reg.coef.iloc[::-1]
    labels = [meta[k].label for k in c.index]
    fig = go.Figure()
    for k, r in c.iterrows():
        sig = abs(r["t"]) >= 2
        fig.add_bar(
            y=[meta[k].label],
            x=[r["impact"]],
            orientation="h",
            marker=_bar_style(meta[k]) | dict(opacity=1.0 if sig else 0.45),
            error_x=dict(type="data", array=[1.96 * r["se"] * r["sd"]], color=MUTED, thickness=1.5),
            text=[f"{r['impact']:+.2f}%  t {r['t']:.1f}"],
            textposition="outside",
            cliponaxis=False,
            customdata=[[r["sd"], r["coef"], r["p"]]],
            hovertemplate="%{y}<br>Typical move %{customdata[0]:.2f}pp → "
            + pair
            + " %{x:+.2f}%<br>Per +1pp: %{customdata[1]:+.2f}%"
            "<br>p = %{customdata[2]:.3f}<extra></extra>",
            showlegend=False,
        )
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_layout(
        xaxis_title=f"{pair} % (+ = USD stronger)",
        yaxis=dict(categoryorder="array", categoryarray=labels, showgrid=False),
        bargap=0.35,
        hovermode="closest",
    )
    show(fig, height=max(260, 70 * len(c) + 90))
    st.caption(
        "Solid bars: |t| ≥ 2 (reliable at about 5%); faded: not distinguishable from zero. "
        + (
            "Hatched bars are the foreign side. "
            if any(m.side == "fx" for m in meta.values())
            else ""
        )
    )


def _coef_table(reg: Regression, meta: dict, drivers: dict, pair: str) -> None:
    rows = []
    for k, r in reg.coef.iterrows():
        m = meta[k]
        d = drivers[m.driver]
        measure = {"gap": f"{d.us_label} − {d.fx_label}", "us": d.us_label, "fx": d.fx_label}
        rows.append(
            {
                "Driver": m.label,
                f"{pair} % per +1pp": r["coef"],
                "Typical move (pp)": r["sd"],
                f"{pair} % per typical move": r["impact"],
                "t-stat": r["t"],
                "p-value": r["p"],
                "Measure": measure[m.side],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            f"{pair} % per +1pp": st.column_config.NumberColumn(format="%+.2f"),
            "Typical move (pp)": st.column_config.NumberColumn(format="%.2f"),
            f"{pair} % per typical move": st.column_config.NumberColumn(format="%+.2f"),
            "t-stat": st.column_config.NumberColumn(format="%.1f"),
            "p-value": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        f"Intercept {reg.const:+.2f}% a quarter: the average {pair} drift not explained by "
        "the drivers."
    )


def _verdict(reg: Regression, meta: dict, pair: str, lead: bool) -> None:
    when = "the next quarter" if lead else "the same quarter"
    sig = reg.coef[reg.coef["t"].abs() >= 2].sort_values("t", key=abs, ascending=False)
    if sig.empty:
        st.markdown(
            f"**Reading:** none of the drivers moves {pair} reliably in {when} over this sample "
            f"(R² {reg.r2:.2f}). Quarterly macro changes explain little of this pair here; "
            "try the other specification, a different start date, or another driver set."
        )
        return
    parts = []
    for k, r in sig.iterrows():
        usd = "stronger" if r["impact"] > 0 else "weaker"
        parts.append(
            f"a typical {r['sd']:.2f}pp quarterly rise in the **{meta[k].label}** went "
            f"with {pair} {r['impact']:+.2f}% (USD {usd}, t {r['t']:.1f})"
        )
    st.markdown(
        f"**Reading ({when}):** "
        + "; ".join(parts)
        + f". Together the drivers explain {reg.r2 * 100:.0f}% of the quarterly variation."
    )


def _contributions(reg: Regression, meta: dict, pair: str) -> None:
    st.markdown(f"**What moved {pair} each quarter** · fitted contribution of each driver")
    n = st.slider("Quarters shown", 4, min(40, reg.nobs), min(12, reg.nobs), key="reg_quarters")
    contrib = reg.contributions().iloc[-n:]
    actual = reg.y.iloc[-n:]
    other = actual - contrib.sum(axis=1)
    x = [str(i.to_period("Q")) for i in contrib.index]
    fig = go.Figure()
    for k in contrib.columns:
        fig.add_bar(
            x=x,
            y=contrib[k],
            name=meta[k].label,
            marker=_bar_style(meta[k]) | dict(line=dict(color="#000000", width=1)),
            hovertemplate="%{x}<br>" + meta[k].label + " %{y:+.2f}%<extra></extra>",
        )
    fig.add_bar(
        x=x,
        y=other,
        name="Intercept + unexplained",
        marker=dict(color=NEUTRAL, line=dict(color="#000000", width=1)),
        hovertemplate="%{x}<br>Intercept + unexplained %{y:+.2f}%<extra></extra>",
    )
    fig.add_scatter(
        x=x,
        y=actual,
        mode="markers",
        name=f"Actual {pair} move",
        marker=dict(color=INK, size=10, symbol="diamond", line=dict(color="#000000", width=2)),
        hovertemplate="%{x}<br>Actual " + pair + " %{y:+.2f}%<extra></extra>",
    )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_layout(
        barmode="relative",
        bargap=0.3,
        yaxis_title=f"{pair} % (+ = USD stronger)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    show(fig, height=440)
    st.caption(
        "Each bar segment is coefficient × that quarter's change in the driver. The segments, "
        "including the grey intercept-plus-unexplained part, add up to the actual move (◆)."
    )


def _actual_vs_fitted(reg: Regression, pair: str) -> None:
    with st.expander(f"Actual vs fitted {pair} quarterly moves"):
        fig = go.Figure()
        fig.add_scatter(
            x=reg.y.index,
            y=reg.y,
            mode="lines",
            name="Actual",
            line=dict(color=INK, width=2),
            hovertemplate="%{x|%Y Q%q}<br>Actual %{y:+.2f}%<extra></extra>",
        )
        fig.add_scatter(
            x=reg.fitted.index,
            y=reg.fitted,
            mode="lines",
            name="Fitted by the drivers",
            line=dict(color=SERIES[0], width=2),
            hovertemplate="%{x|%Y Q%q}<br>Fitted %{y:+.2f}%<extra></extra>",
        )
        fig.add_hline(y=0, line=dict(color=AXIS, width=1))
        fig.update_layout(yaxis_title=f"{pair} quarterly %", hovermode="x unified")
        show(fig, height=360)


def _rolling(reg: Regression, meta: dict, pair: str) -> None:
    st.markdown("**Is the relationship stable?** · rolling-window coefficients")
    hi = reg.nobs - 4
    if hi < 16:
        st.info("Too few quarters for rolling windows.")
        return
    window = st.slider("Window (quarters)", 16, min(40, hi), min(20, hi), 2, key="reg_window")
    roll = rolling_ols(reg.y, reg.X, window)
    if roll["coef"].empty:
        st.info("Too few quarters for rolling windows.")
        return
    impact = roll["coef"] * reg.coef["sd"]  # full-sample typical move, so lines stay comparable
    fig = go.Figure()
    for k in impact.columns:
        m = meta[k]
        fig.add_scatter(
            x=impact.index,
            y=impact[k],
            mode="lines",
            name=m.label,
            line=dict(color=COLORS[m.driver], width=2, dash="dot" if m.side == "fx" else "solid"),
            customdata=roll["t"][k],
            hovertemplate="%{x|%Y Q%q}<br>"
            + m.label
            + " %{y:+.2f}% (t %{customdata:.1f})<extra></extra>",
        )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_layout(
        yaxis_title=f"{pair} % per typical move",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    show(fig, height=400)
    st.caption(
        f"Each point is the model refitted on the {window} quarters ending there. Lines that "
        "cross zero or swing widely mean the driver's effect has not been stable."
        + (
            " Dotted lines are the foreign side."
            if any(m.side == "fx" for m in meta.values())
            else ""
        )
    )


def _all_g10(macro, pairs, us_extra, fed_target, load_country, chosen, spec, lead, start_ts):
    st.markdown("**Every G10 currency, same model**")
    if not st.toggle(
        "Run for all 11 economies",
        value=False,
        key="reg_all",
        help="Loads every economy's data the first time (cached afterwards).",
    ):
        st.caption("Switch on to compare the drivers across all G10 currencies.")
        return
    rows, cells = [], {}
    for code, country in COUNTRIES.items():
        pair = f"USD{country.currency}"
        px = pairs.get(pair)
        if px is None:
            continue
        drivers, _, _ = load_drivers(macro, fed_target, us_extra, country, load_country(code))
        X, meta = design(drivers, chosen, spec, country)
        reg = fit(response(px, lead), X, start_ts)
        rate = drivers["RATE"].name if "RATE" in drivers else "n/a"
        label = f"{pair} · {country.name}" + (
            " (3M)" if rate == "3M rate" and "RATE" in chosen else ""
        )
        if reg is None:
            rows.append({"Pair": label, "Quarters": 0})
            continue
        top = reg.coef["t"].abs().idxmax()
        rows.append(
            {
                "Pair": label,
                "R²": reg.r2,
                "Adjusted R²": reg.adj_r2,
                "Model p-value": reg.f_pvalue,
                "Strongest driver": meta[top].generic,
                "Quarters": reg.nobs,
            }
        )
        cells[label] = {meta[k].generic: (r["t"], r["impact"]) for k, r in reg.coef.iterrows()}
    if not cells:
        st.info("No economy had enough data.")
        return
    cols = list(dict.fromkeys(c for v in cells.values() for c in v))
    ys = list(cells)
    t = np.array([[cells[p].get(c, (np.nan, np.nan))[0] for c in cols] for p in ys])
    imp = np.array([[cells[p].get(c, (np.nan, np.nan))[1] for c in cols] for p in ys])
    text = [
        [
            f"{imp[i, j]:+.2f}%<br>t {t[i, j]:.1f}" if np.isfinite(t[i, j]) else ""
            for j in range(len(cols))
        ]
        for i in range(len(ys))
    ]
    fig = go.Figure(
        go.Heatmap(
            z=np.clip(t, -4, 4),
            x=cols,
            y=ys,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color=INK, size=12),
            customdata=np.dstack([t, imp]),
            colorscale=DIVERGING,
            zmin=-4,
            zmax=4,
            xgap=3,
            ygap=3,
            colorbar=dict(title="t-stat", thickness=10, outlinewidth=0),
            hovertemplate="%{y}<br>%{x}<br>%{customdata[1]:+.2f}% per typical move"
            "<br>t = %{customdata[0]:.1f}<extra></extra>",
            hoverongaps=False,
        )
    )
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(plot_bgcolor=NEUTRAL)
    show(fig, height=90 + 42 * len(ys))
    st.caption(
        "Cell text: USD move per typical quarterly move in the driver, and its t-stat. Colour "
        "is the t-stat: blue = the driver rising goes with a stronger dollar, red = weaker, "
        "grey = no reliable link. (3M) = no 2Y yield available, so the 3-month rate is used."
    )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "R²": st.column_config.NumberColumn(format="%.2f"),
            "Adjusted R²": st.column_config.NumberColumn(format="%.2f"),
            "Model p-value": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def _data(reg: Regression, meta: dict, pair: str, code: str) -> None:
    with st.expander("Regression data"):
        table = pd.DataFrame({f"{pair} % (response)": reg.y})
        for k in reg.X.columns:
            table[f"Δ {meta[k].label} (pp)"] = reg.X[k]
        table["Fitted %"] = reg.fitted
        table["Residual %"] = reg.resid
        table = table.sort_index(ascending=False)
        table.index = table.index.to_period("Q").astype(str)
        table.index.name = "Quarter"
        st.dataframe(table, width="stretch")
        st.download_button(
            "Download CSV",
            table.to_csv().encode(),
            file_name=f"regression_usd{code.lower()}.csv",
            mime="text/csv",
            key="reg_csv",
        )
