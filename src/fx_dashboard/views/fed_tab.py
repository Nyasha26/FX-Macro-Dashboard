"""Tab 5: USD returns against each currency and the broad dollar index in Fed hiking,
on-hold and cutting regimes."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fx_dashboard.analysis import REGIME_ORDER, episode_returns, regime_blocks, regime_stats
from fx_dashboard.series import FX_PAIRS
from fx_dashboard.theme import (
    AXIS,
    DIVERGING,
    INK,
    NEUTRAL,
    REGIME_COLORS,
    SURFACE,
    regime_mark,
    regime_name,
    show,
)
from fx_dashboard.views.regime_controls import (
    build_fed_regime,
    definitions,
    regime_controls,
    response_returns,
)

BENCHMARK = "Broad USD index"
SYMBOLS = {"Up": "triangle-up", "Flat": "square", "Down": "triangle-down"}
MEASURES = {"Average quarterly %": "mean", "Annualised %": "ann"}


def _name(reg: str) -> str:
    return regime_name("FEDFUNDS", reg)


def _mark(reg: str) -> str:
    return regime_mark("FEDFUNDS", reg)


def render(
    pairs: dict[str, pd.Series],
    dollar: pd.Series,
    macro: dict[str, pd.Series],
    start,
    fed_target: pd.Series | None,
) -> None:
    st.subheader("The dollar in Fed hiking and cutting cycles")
    st.caption(
        "USD return against every currency, and the broad dollar index, while the Fed is "
        "hiking, on hold or cutting. Quarterly, from "
        f"{pd.Timestamp(start):%b %Y} (sidebar). Pairs are foreign currency per USD: "
        "positive = USD stronger."
    )

    settings = regime_controls("fed", subject="USD", fed_only=True)
    if fed_target is None:
        st.warning(
            "The Fed target rate failed to load, so regimes use the effective fed funds rate "
            "with a 10bp band instead. Press Refresh data to retry."
        )
    start_ts = pd.Timestamp(start)
    fed = build_fed_regime(macro, fed_target, settings)
    fed = fed[fed.index >= start_ts].dropna()
    if fed.empty:
        st.warning("No Fed regime data in the selected sample.")
        return

    names = [n for n in FX_PAIRS if n in pairs]
    series = {BENCHMARK: dollar} | {n: pairs[n] for n in names}
    returns = {n: response_returns(s, settings, start) for n, s in series.items()}
    regimes = fed.to_frame("FEDFUNDS")
    stats = pd.concat(
        [regime_stats(r, regimes).assign(pair=n) for n, r in returns.items()], ignore_index=True
    )

    _tiles(fed, fed_target, macro)
    _timeline(dollar, fed_target, macro, fed, start_ts)
    _dollar_table(stats)
    _pairs_chart(stats, names)
    _episodes(returns, fed)
    runs = [r for *_, r in regime_blocks(fed)]
    st.caption(
        f"Definition: {definitions(settings)['FEDFUNDS']}. Moves are dated by quarter-end "
        f"target. Averages describe history, not forecasts. The sample holds only "
        f"{runs.count('Up')} hiking and {runs.count('Down')} cutting runs, so one episode "
        "(for example the 2008 crisis cuts) can dominate a regime's average."
    )


def _tiles(fed: pd.Series, fed_target: pd.Series | None, macro) -> None:
    blocks = regime_blocks(fed)
    now_start, _, now = blocks[-1]
    cols = st.columns(5)
    cols[0].metric(
        "Fed regime now",
        _mark(now),
        help=f"Latest complete quarter {fed.index[-1].to_period('Q')}; "
        f"this regime began {now_start.to_period('Q')}.",
    )
    rate = fed_target if fed_target is not None else macro["FEDFUNDS"]
    label = "Fed target (upper bound)" if fed_target is not None else "Fed funds (effective)"
    year_ago = rate.asof(rate.index[-1] - pd.DateOffset(years=1))
    cols[1].metric(
        label,
        f"{rate.iloc[-1]:.2f}%",
        delta=f"{(rate.iloc[-1] - year_ago) * 100:+.0f}bp vs a year ago",
        delta_color="off",
        delta_arrow="off",
        help=f"As of {rate.index[-1]:%d %b %Y}",
    )
    counts = fed.value_counts()
    for col, reg in zip(cols[2:], REGIME_ORDER, strict=True):
        col.metric(f"Quarters {_name(reg).lower()}", int(counts.get(reg, 0)))


def _shade(fig: go.Figure, fed: pd.Series) -> None:
    for s, e, reg in regime_blocks(fed):
        if reg != "Flat":
            fig.add_vrect(
                x0=s,
                x1=e,
                fillcolor=REGIME_COLORS[reg],
                opacity=0.25,
                line_width=0,
                layer="below",
                row="all",
                col=1,
            )


def _timeline(dollar, fed_target, macro, fed: pd.Series, start_ts) -> None:
    st.markdown("**Broad dollar index and the Fed's policy rate** · hiking and cutting shaded")
    level = dollar[dollar.index >= start_ts].resample("W-FRI").last().dropna()
    rate = fed_target if fed_target is not None else macro["FEDFUNDS"]
    rate = rate[rate.index >= start_ts].resample("W-FRI").last().dropna()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06
    )
    fig.add_scatter(
        x=level.index,
        y=level.values,
        mode="lines",
        name=BENCHMARK,
        line=dict(color=INK, width=2),
        hovertemplate="%{x|%d %b %Y}<br>USD index %{y:.2f}<extra></extra>",
        showlegend=False,
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=rate.index,
        y=rate.values,
        mode="lines",
        name="Fed target",
        line=dict(color=INK, width=2, shape="hv"),
        hovertemplate="%{x|%d %b %Y}<br>Fed target %{y:.2f}%<extra></extra>",
        showlegend=False,
        row=2,
        col=1,
    )
    _shade(fig, fed)
    for reg in ("Up", "Down"):  # legend entries for the shading
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"Fed {_name(reg).lower()}",
            marker=dict(symbol="square", size=12, color=REGIME_COLORS[reg], opacity=0.5),
            row=1,
            col=1,
        )
    fig.update_yaxes(title_text="Broad USD index", row=1, col=1)
    fig.update_yaxes(title_text="Fed target %", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    show(fig, height=520)


def _dollar_table(stats: pd.DataFrame) -> None:
    st.markdown("**Broad dollar index in each Fed regime**")
    d = stats[stats["pair"] == BENCHMARK].set_index("regime")
    cols = st.columns(len(d))
    for col, reg in zip(cols, [r for r in REGIME_ORDER if r in d.index], strict=True):
        row = d.loc[reg]
        col.metric(
            f"USD while {_name(reg).lower()}",
            f"{row['ann']:+.1f}% a year",
            delta=f"{row['mean']:+.2f}% avg quarter · up in {row['hit_rate']:.0f}% of "
            f"{int(row['n'])} qtrs",
            delta_color="off",
            delta_arrow="off",
        )
    if {"Up", "Down"} <= set(d.index):
        t = _tstat(d.loc["Up"], d.loc["Down"])
        gap = d.loc["Up", "mean"] - d.loc["Down", "mean"]
        verdict = (
            "a gap that is statistically distinguishable from zero"
            if abs(t) >= 2
            else "a gap too small to separate from noise with this few quarters"
        )
        st.caption(
            f"The broad dollar averaged {d.loc['Up', 'mean']:+.2f}% a quarter while the Fed "
            f"was hiking and {d.loc['Down', 'mean']:+.2f}% while it was cutting: "
            f"{gap:+.2f}pp a quarter (t = {t:.1f}), {verdict}."
        )


def _tstat(a: pd.Series, b: pd.Series) -> float:
    """Welch t-statistic for the difference in mean quarterly return between two regimes."""
    se = np.sqrt(a["std"] ** 2 / a["n"] + b["std"] ** 2 / b["n"])
    return float((a["mean"] - b["mean"]) / se) if se > 0 else np.nan


def _pairs_chart(stats: pd.DataFrame, names: list[str]) -> None:
    st.markdown("**USD against each currency by Fed regime**")
    label = st.radio(
        "Measure", list(MEASURES), horizontal=True, key="fed_measure", label_visibility="collapsed"
    )
    m = MEASURES[label]
    wide = stats.pivot(index="pair", columns="regime", values=m)
    gap = (wide.get("Up") - wide.get("Down")) if {"Up", "Down"} <= set(wide.columns) else None
    order = [BENCHMARK] + (
        list(gap.drop(BENCHMARK).sort_values(ascending=False).index) if gap is not None else names
    )

    fig = go.Figure()
    # Thin connector from cutting to hiking shows the spread for each pair.
    if gap is not None:
        for p in order:
            fig.add_scatter(
                x=[wide.loc[p, "Down"], wide.loc[p, "Up"]],
                y=[p, p],
                mode="lines",
                line=dict(color=AXIS, width=3),
                hoverinfo="skip",
                showlegend=False,
            )
    for reg in REGIME_ORDER:
        d = stats[stats["regime"] == reg].set_index("pair").reindex(order).dropna(subset=[m])
        if d.empty:
            continue
        fig.add_scatter(
            x=d[m],
            y=d.index,
            mode="markers",
            name=_name(reg),
            marker=dict(
                symbol=SYMBOLS[reg],
                size=13,
                color=REGIME_COLORS[reg],
                line=dict(color=SURFACE, width=1),
            ),
            customdata=d[["mean", "ann", "hit_rate", "n"]].to_numpy(),
            hovertemplate=(
                "%{y} · Fed " + _name(reg).lower() + "<br>Avg quarter %{customdata[0]:+.2f}%"
                "<br>Annualised %{customdata[1]:+.1f}%"
                "<br>USD up in %{customdata[2]:.0f}% of quarters"
                "<br>n = %{customdata[3]}<extra></extra>"
            ),
        )
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_layout(
        xaxis_title=f"{label} move in USD/ccy (+ = USD stronger)",
        yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=order),
        hovermode="closest",
    )
    show(fig, height=140 + 40 * len(order))
    st.caption(
        "Broad USD index first, then pairs sorted by how much better the dollar did against "
        "that currency when the Fed was hiking than when it was cutting."
    )

    table = []
    for p in order:
        s = stats[stats["pair"] == p].set_index("regime")
        row = {"Pair": p}
        for reg in REGIME_ORDER:
            row[f"{_name(reg)} %"] = s[m].get(reg, np.nan)
        if {"Up", "Down"} <= set(s.index):
            row["Hiking − cutting"] = s.loc["Up", m] - s.loc["Down", m]
            row["t-stat"] = _tstat(s.loc["Up"], s.loc["Down"])
        for reg in ("Up", "Down"):
            row[f"USD up, {_name(reg).lower()} (% qtrs)"] = s["hit_rate"].get(reg, np.nan)
        row["Quarters (hike/hold/cut)"] = " / ".join(
            str(int(s["n"].get(r, 0))) for r in REGIME_ORDER
        )
        table.append(row)
    table = pd.DataFrame(table)
    pct = st.column_config.NumberColumn(format="%+.2f")
    hit = st.column_config.NumberColumn(format="%.0f")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            **{f"{_name(r)} %": pct for r in REGIME_ORDER},
            "Hiking − cutting": pct,
            "t-stat": st.column_config.NumberColumn(
                format="%.1f",
                help="Welch t-statistic on the hiking vs cutting difference in average "
                "quarterly return. |t| above about 2 is unlikely to be chance.",
            ),
            "USD up, hiking (% qtrs)": hit,
            "USD up, cutting (% qtrs)": hit,
        },
    )
    st.download_button(
        "Download CSV",
        table.to_csv(index=False).encode(),
        "usd_by_fed_regime.csv",
        "text/csv",
        key="fed_csv",
    )


def _episodes(returns: dict[str, pd.Series], fed: pd.Series) -> None:
    st.markdown("**Cycle by cycle** · cumulative % USD move over each Fed regime")
    show_hold = st.toggle("Include on-hold periods", value=False, key="fed_show_hold")
    ep = episode_returns(returns, fed)
    if not show_hold:
        ep = ep[ep["regime"] != "Flat"]
    if ep.empty:
        st.info("No hiking or cutting cycles in the selected sample.")
        return
    cols = list(returns)
    z = ep[cols].to_numpy(dtype=float)
    labels = [
        f"{_mark(r)} {s.to_period('Q')}–{e.to_period('Q')} ({q}q)"
        for s, e, r, q in zip(ep["start"], ep["end"], ep["regime"], ep["quarters"], strict=True)
    ]
    text = [[f"{v:+.1f}" if np.isfinite(v) else "" for v in row] for row in z]
    lim = max(np.nanpercentile(np.abs(z), 95) if np.isfinite(z).any() else 1, 1)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=cols,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color=INK, size=12),
            colorscale=DIVERGING,
            zmin=-lim,
            zmax=lim,
            xgap=2,
            ygap=2,
            colorbar=dict(title="Cum. %", thickness=10, outlinewidth=0),
            hovertemplate="%{y}<br>%{x}: %{z:+.1f}% over the period<extra></extra>",
            hoverongaps=False,
        )
    )
    fig.add_vline(x=0.5, line=dict(color=SURFACE, width=6))  # set the benchmark apart
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(plot_bgcolor=NEUTRAL)
    show(fig, height=130 + 38 * len(labels))
    st.caption(
        "Each row is one unbroken run of a regime; blue = USD gained against that currency "
        "over the run, red = USD lost. Blank cells: the pair's history does not cover the "
        "whole run. The colour scale is capped so one extreme cycle does not wash out the rest."
    )
