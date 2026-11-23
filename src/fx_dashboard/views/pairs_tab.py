"""Tab 3: toggleable USD FX pairs with a macro shock scenario."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fx_dashboard.analysis import (
    DRIVERS,
    fit_sensitivity,
    macro_drivers,
    quarterly_returns,
    rebase,
    scenario_impact,
    trailing_returns,
)
from fx_dashboard.series import FX_PAIRS, MACRO_VARS
from fx_dashboard.theme import AXIS, DRIVER_COLORS, INK, SERIES, SURFACE, show

BENCHMARK = "Broad USD index"

# key, label, unit, min, max, step, multiplier to model units (pp or %)
SHOCKS = [
    ("FEDFUNDS", "Fed funds", "bps", -300, 300, 2, 0.01),
    ("DGS2", "2Y Treasury yield", "bps", -300, 300, 2, 0.01),
    ("CPI", "CPI inflation, q/q", "pp", -3.0, 3.0, 0.1, 1.0),
    ("GDP", "Real GDP, q/q", "%", -10.0, 10.0, 1.0, 1.0),
    ("T10Y2Y", "10Y–2Y spread", "bps", -300, 300, 2, 0.01),
]
SHORT = {k: MACRO_VARS[k].short for k in DRIVERS}
BETA_UNITS = {
    "FEDFUNDS": "per +100bp",
    "DGS2": "per +100bp",
    "CPI": "per +1pp",
    "GDP": "per +1%",
    "T10Y2Y": "per +100bp",
}


def _reset_shocks() -> None:
    for key, *_rest, step, _mult in SHOCKS:
        st.session_state[f"shock_{key}"] = type(step)(0)


def render(
    pairs: dict[str, pd.Series], dollar: pd.Series, macro: dict[str, pd.Series], start
) -> None:
    st.subheader("USD FX pairs")
    st.caption(
        "All pairs are quoted as foreign currency per 1 USD, so up = stronger dollar. "
        "Switch pairs on or off."
    )

    available = [p for p in FX_PAIRS if p in pairs]
    toggle_cols = st.columns(len(available))
    selected = [
        name
        for col, name in zip(toggle_cols, available, strict=True)
        if col.toggle(name, value=True, key=f"pair_on_{name}", help=FX_PAIRS[name].note or None)
    ]
    include_bench = st.toggle(f"Include {BENCHMARK} as a benchmark", value=True, key="pair_bench")

    if not selected:
        st.info("All pairs are switched off. Toggle one on above.")
        return

    series = {n: pairs[n] for n in selected}
    if include_bench:
        series[BENCHMARK] = dollar

    _snapshot(series)
    _performance(series, start)
    _scenario(series, macro, start)


def _snapshot(series: dict[str, pd.Series]) -> None:
    rows = []
    for name, s in series.items():
        tr = trailing_returns(s)
        rows.append({"Pair": name, "Spot": s.iloc[-1], "As of": s.index[-1].date(), **tr})
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Spot": st.column_config.NumberColumn(format="%.4f"),
            **{
                k: st.column_config.NumberColumn(f"{k} %", format="%+.2f")
                for k in ("1M", "3M", "YTD", "1Y")
            },
        },
    )


def _performance(series: dict[str, pd.Series], start) -> None:
    st.markdown(f"**Performance since {pd.Timestamp(start):%b %Y}** · rebased to 100, weekly")
    long = pd.concat(
        [
            rebase(s.resample("W-FRI").last(), start).rename("value").to_frame().assign(pair=n)
            for n, s in series.items()
        ]
    ).reset_index()
    n_cols = 3
    n_rows = -(-len(series) // n_cols)
    fig = px.line(
        long,
        x="date",
        y="value",
        facet_col="pair",
        facet_col_wrap=n_cols,
        facet_row_spacing=0.12 / max(n_rows - 1, 1) if n_rows > 1 else 0.0,
        facet_col_spacing=0.06,
        color_discrete_sequence=[SERIES[0]],
        category_orders={"pair": list(series)},
    )
    fig.update_traces(line_width=2, hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}<extra></extra>")
    fig.update_yaxes(matches=None, showticklabels=True, title="")
    fig.update_xaxes(title="", showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))
    fig.add_hline(y=100, line=dict(color=AXIS, width=1), row="all", col="all")
    fig.update_layout(showlegend=False, margin=dict(t=30))
    show(fig, height=40 + 230 * n_rows)


def _scenario(series: dict[str, pd.Series], macro: dict[str, pd.Series], start) -> None:
    st.markdown("#### Macro shock scenario")
    st.caption(
        "Move each variable up or down with the − / + buttons, or type a value. The model "
        "regresses each pair's quarterly % change on the same quarter's change in fed funds, "
        "the 2Y yield, CPI, real GDP and the 10Y–2Y spread, then applies your shocks to the "
        "fitted sensitivities."
    )

    cols = st.columns(len(SHOCKS) + 1)
    shocks: dict[str, float] = {}
    for col, (key, label, unit, lo, hi, step, mult) in zip(cols, SHOCKS, strict=False):
        # Default lives in session state (not value=) so the Reset button can set it.
        st.session_state.setdefault(f"shock_{key}", type(step)(0))
        value = col.number_input(
            f"{label} ({unit})",
            min_value=lo,
            max_value=hi,
            step=step,
            format="%d" if isinstance(step, int) else "%.1f",
            key=f"shock_{key}",
        )
        shocks[key] = value * mult
    cols[-1].markdown("&nbsp;")
    cols[-1].button("Reset shocks", on_click=_reset_shocks, width="stretch")

    drivers = macro_drivers(macro)
    start_ts = pd.Timestamp(start)
    results, skipped = [], []
    for name, s in series.items():
        ret = quarterly_returns(s)
        sens = fit_sensitivity(ret[ret.index >= start_ts], drivers)
        if sens is None:
            skipped.append(name)
            continue
        contrib, total, se = scenario_impact(sens, shocks)
        results.append((name, s.iloc[-1], sens, contrib, total, se))

    if skipped:
        st.warning(f"Not enough quarterly history since the start date for: {', '.join(skipped)}")
    if not results:
        return

    any_shock = any(v != 0 for v in shocks.values())
    if any_shock:
        _impact_chart(results)
    else:
        st.info("All shocks are zero. Move a variable above to see the implied FX moves.")

    impact = pd.DataFrame(
        [
            {
                "Pair": name,
                "Spot": spot,
                "Implied move %": total,
                "95% low %": total - 1.96 * se,
                "95% high %": total + 1.96 * se,
                "Implied spot": spot * (1 + total / 100),
                "R²": sens.r2,
                "Quarters": sens.nobs,
            }
            for name, spot, sens, _c, total, se in results
        ]
    )
    pct_fmt = st.column_config.NumberColumn(format="%+.2f")
    st.dataframe(
        impact,
        hide_index=True,
        width="stretch",
        column_config={
            "Spot": st.column_config.NumberColumn(format="%.4f"),
            "Implied spot": st.column_config.NumberColumn(format="%.4f"),
            "Implied move %": pct_fmt,
            "95% low %": pct_fmt,
            "95% high %": pct_fmt,
            "R²": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    with st.expander("Fitted sensitivities (% move in the pair)"):
        betas = pd.DataFrame({name: sens.beta for name, _s, sens, *_ in results}).T.rename(
            columns=lambda k: f"{SHORT[k]} {BETA_UNITS[k]}"
        )
        tstats = pd.DataFrame({name: sens.tstat for name, _s, sens, *_ in results}).T.rename(
            columns=lambda k: f"{SHORT[k]} t-stat"
        )
        st.dataframe(
            pd.concat([betas, tstats], axis=1).round(2),
            width="stretch",
        )
        st.caption(
            "Robust (HC1) standard errors. |t| above about 2 suggests the relationship is "
            "distinguishable from zero. Fed funds and the 2Y yield move together, and the "
            "10Y–2Y spread often falls when the 2Y yield rises, so their separate coefficients "
            "are noisy; a combined shock that mirrors how they usually move is more reliable. "
            "A 2Y shock with the spread held fixed means the 10Y yield moves by the same amount."
        )

    st.caption(
        "These are historical, same-quarter associations, not forecasts or causal effects. "
        "Implied spot applies the move to the latest rate. Shocks are measured against a "
        "zero-change baseline: CPI and GDP shocks are changes in the quarter-on-quarter "
        "rate."
    )


def _impact_chart(results) -> None:
    st.markdown("**Implied % move by pair**, split by driver · total with 95% band")
    names = [r[0] for r in results][::-1]  # first pair at the top
    by_name = {r[0]: r for r in results}
    fig = go.Figure()
    for key in DRIVERS:
        vals = [by_name[n][3][key] for n in names]
        if not any(vals):
            continue
        fig.add_bar(
            y=names,
            x=vals,
            orientation="h",
            name=SHORT[key],
            marker=dict(color=DRIVER_COLORS[key], line=dict(color=SURFACE, width=2)),
            hovertemplate="%{y} · " + SHORT[key] + ": %{x:+.2f}%<extra></extra>",
        )
    fig.add_scatter(
        y=names,
        x=[by_name[n][4] for n in names],
        mode="markers",
        name="Total (95% band)",
        marker=dict(symbol="diamond", size=11, color=INK, line=dict(color=SURFACE, width=2)),
        error_x=dict(
            type="data",
            array=[1.96 * by_name[n][5] for n in names],
            color=INK,
            thickness=1.5,
            width=4,
        ),
        hovertemplate="%{y} · total %{x:+.2f}%<extra></extra>",
    )
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_layout(
        barmode="relative", xaxis_title="% move (+ = USD stronger)", hovermode="closest"
    )
    show(fig, height=110 + 38 * len(names))
