"""Tab 2: Nominal broad dollar index regime analysis."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fx_dashboard.analysis import (
    DRIVERS,
    REGIME_ORDER,
    regime_blocks,
    regime_matrix,
    regime_stats,
    trailing_returns,
)
from fx_dashboard.theme import (
    AXIS,
    DIVERGING,
    INK,
    REGIME_COLORS,
    REGIME_MARKS,
    regime_mark,
    regime_name,
    show,
)
from fx_dashboard.views.regime_controls import (
    SHORT,
    build_regimes,
    definitions,
    regime_controls,
    response_returns,
)


def render(
    dollar: pd.Series, macro: dict[str, pd.Series], start, fed_target: pd.Series | None = None
) -> None:
    st.subheader("Nominal broad dollar index: regime analysis")
    st.caption(
        "How the Fed's nominal broad USD index (DTWEXBGS, Jan 2006 = 100) behaves when each "
        "macro variable is going up or down. Quarterly, because GDP is quarterly."
    )

    settings = regime_controls("dxy", subject="Dollar")
    response = settings.response
    start_ts = pd.Timestamp(start)
    regimes = build_regimes(macro, settings, start, fed_target)
    ret = response_returns(dollar, settings, start)
    stats = regime_stats(ret, regimes)

    if stats.empty:
        st.warning("No overlapping quarters in the selected sample.")
        return

    # --- headline tiles: index level and current regimes ---
    tr = trailing_returns(dollar)
    tiles = st.columns(len(DRIVERS) + 1)
    tiles[0].metric(
        "Broad USD index",
        f"{dollar.iloc[-1]:.2f}",
        delta=f"{tr['1Y']:+.1f}% 1y",
        delta_color="off",
        help=f"Latest: {dollar.index[-1]:%d %b %Y}",
    )
    for col, k in zip(tiles[1:], DRIVERS, strict=True):
        last = regimes[k].dropna()
        if last.empty:
            col.metric(SHORT[k], "n/a")
            continue
        q = last.index[-1].to_period("Q")
        col.metric(
            f"{SHORT[k]} regime", regime_mark(k, last.iloc[-1]), help=f"Latest complete: {q}"
        )

    # --- average dollar move by regime ---
    st.markdown(f"**Average quarterly dollar move by regime** · {response.lower()}")
    fig = go.Figure()
    for reg in REGIME_ORDER:
        d = stats[stats["regime"] == reg]
        if d.empty:
            continue
        fig.add_bar(
            x=d["variable"].map(SHORT),
            y=d["mean"],
            name=f"{REGIME_MARKS[reg]} (Fed {regime_name('FEDFUNDS', reg).lower()})",
            marker=dict(color=REGIME_COLORS[reg]),
            customdata=d[["median", "hit_rate", "n"]]
            .assign(label=[regime_mark(v, reg) for v in d["variable"]])
            .to_numpy(dtype=object),
            hovertemplate=(
                "%{x} · %{customdata[3]}<br>Mean %{y:+.2f}%<br>Median %{customdata[0]:+.2f}%"
                "<br>Dollar up in %{customdata[1]:.0f}% of quarters"
                "<br>n = %{customdata[2]}<extra></extra>"
            ),
        )
    fig.update_layout(barmode="group", yaxis_title="Mean % change", hovermode="closest")
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    show(fig, height=380)

    defs = definitions(settings)
    table = stats.assign(
        variable=stats["variable"].map(SHORT),
        definition=stats["variable"].map(defs),
        regime=[regime_mark(v, r) for v, r in zip(stats["variable"], stats["regime"], strict=True)],
    )[["variable", "regime", "definition", "mean", "median", "hit_rate", "n"]]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "variable": "Variable",
            "regime": "Regime",
            "definition": "Definition",
            "mean": st.column_config.NumberColumn("Mean %", format="%+.2f"),
            "median": st.column_config.NumberColumn("Median %", format="%+.2f"),
            "hit_rate": st.column_config.NumberColumn("Dollar up (% of qtrs)", format="%.0f"),
            "n": st.column_config.NumberColumn("Quarters"),
        },
    )

    # --- two-variable regime matrix ---
    st.markdown(
        "**Combined regimes**: average quarterly dollar move when two variables move together"
    )
    m1, m2 = st.columns(2)
    a = m1.selectbox("Rows", DRIVERS, index=0, format_func=SHORT.get, key="dxy_a")
    b_options = [d for d in DRIVERS if d != a]
    b = m2.selectbox(
        "Columns",
        b_options,
        index=b_options.index("GDP") if "GDP" in b_options else 0,
        format_func=SHORT.get,
        key="dxy_b",
    )
    mean, count = regime_matrix(ret, regimes, a, b)
    if mean.empty:
        st.info("No overlapping quarters for this combination.")
    else:
        text = [
            [
                f"{mean.iat[i, j]:+.2f}%<br>n={count.iat[i, j]}" if count.iat[i, j] else "–"
                for j in range(mean.shape[1])
            ]
            for i in range(mean.shape[0])
        ]
        lim = max(abs(mean.min().min()), abs(mean.max().max()), 0.5)
        heat = go.Figure(
            go.Heatmap(
                z=mean.to_numpy(),
                x=[f"{SHORT[b]} {regime_mark(b, c)}" for c in mean.columns],
                y=[f"{SHORT[a]} {regime_mark(a, r)}" for r in mean.index],
                text=text,
                texttemplate="%{text}",
                textfont=dict(color=INK, size=14),
                colorscale=DIVERGING,
                zmin=-lim,
                zmax=lim,
                xgap=2,
                ygap=2,
                colorbar=dict(title="Mean %", thickness=10, outlinewidth=0),
                hovertemplate="%{y} & %{x}<br>%{text}<extra></extra>",
            )
        )
        heat.update_yaxes(autorange="reversed", showgrid=False)
        heat.update_xaxes(showgrid=False, side="top")
        show(heat, height=110 + 90 * mean.shape[0])

    # --- timeline with regime shading ---
    st.markdown("**Dollar index through each regime**")
    shade_var = st.selectbox("Shade by", DRIVERS, index=0, format_func=SHORT.get, key="dxy_shade")
    level = dollar[dollar.index >= start_ts].resample("W-FRI").last().dropna()
    tl = go.Figure(
        go.Scatter(
            x=level.index,
            y=level.values,
            mode="lines",
            name="Broad USD index",
            line=dict(color=INK, width=2),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    for s, e, reg in regime_blocks(regimes[shade_var]):
        tl.add_vrect(
            x0=s, x1=e, fillcolor=REGIME_COLORS[reg], opacity=0.3, line_width=0, layer="below"
        )
    for reg in REGIME_ORDER:  # legend entries for the shading
        if (regimes[shade_var] == reg).any():
            tl.add_scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"{SHORT[shade_var]} {regime_mark(shade_var, reg)}",
                marker=dict(symbol="square", size=12, color=REGIME_COLORS[reg], opacity=0.5),
            )
    tl.update_layout(yaxis_title="Index", hovermode="x")
    show(tl, height=380)

    st.caption(
        f"Definitions: {'; '.join(f'{SHORT[k]}: {v}' for k, v in defs.items())}. "
        "Regime statistics describe historical co-movement, not causation; small "
        "cells (low n) are noisy. Blue marks Up (fed funds: hiking) regimes, red Down (cutting), "
        "grey Flat (on hold)."
    )
