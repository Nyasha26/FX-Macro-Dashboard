"""Tab 4: every USD pair across each macro variable's regimes."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fx_dashboard.analysis import DRIVERS, REGIME_ORDER, pair_regime_stats
from fx_dashboard.series import FX_PAIRS
from fx_dashboard.theme import (
    AXIS,
    DIVERGING,
    DOWN,
    INK,
    NEUTRAL,
    SURFACE,
    UP,
    regime_mark,
    show,
)
from fx_dashboard.views.regime_controls import (
    SHORT,
    build_regimes,
    definitions,
    regime_controls,
    regime_phrase,
    response_returns,
)

BENCHMARK = "Broad USD index"


def _ccy(pair: str) -> str:
    return pair[3:] if pair.startswith("USD") else pair


def render(
    pairs: dict[str, pd.Series],
    dollar: pd.Series,
    macro: dict[str, pd.Series],
    start,
    fed_target: pd.Series | None = None,
) -> None:
    st.subheader("Currency pairs across macro regimes")
    st.caption(
        f"Average quarterly % move of every USD pair in each fed funds, 2Y yield, CPI, GDP and "
        f"10Y–2Y spread regime since {pd.Timestamp(start):%b %Y} (set the start date in the "
        "sidebar). Pairs are foreign currency per USD: positive = USD stronger / that currency "
        "weaker."
    )

    settings = regime_controls("pr", subject="FX")
    min_n = st.slider(
        "Hide results based on fewer than this many quarters",
        1,
        20,
        6,
        key="pr_min_n",
        help="Averages over a handful of quarters are mostly noise.",
    )

    names = [n for n in FX_PAIRS if n in pairs]
    series = {n: pairs[n] for n in names} | {BENCHMARK: dollar}
    regimes = build_regimes(macro, settings, start, fed_target)
    stats = pair_regime_stats(
        {n: response_returns(s, settings, start) for n, s in series.items()}, regimes
    )
    if stats.empty:
        st.warning("No overlapping quarters in the selected sample.")
        return
    stats["shown"] = stats["n"] >= min_n

    columns = [
        (v, r)
        for v in DRIVERS
        for r in REGIME_ORDER
        if ((stats["variable"] == v) & (stats["regime"] == r)).any()
    ]
    rows = list(series)

    _heatmap(stats, columns, rows)
    _ranking(stats, series, settings, start, names)
    _summary(stats, columns, names, settings)


def _heatmap(stats: pd.DataFrame, columns, rows) -> None:
    st.markdown("**All pairs × all regimes** · mean quarterly % move")
    idx = stats.set_index(["pair", "variable", "regime"])
    z = np.full((len(rows), len(columns)), np.nan)
    text = [["" for _ in columns] for _ in rows]
    custom = [[[np.nan, np.nan, 0] for _ in columns] for _ in rows]
    for i, p in enumerate(rows):
        for j, (v, r) in enumerate(columns):
            if (p, v, r) not in idx.index:
                continue
            row = idx.loc[(p, v, r)]
            custom[i][j] = [row["median"], row["hit_rate"], int(row["n"])]
            if row["shown"]:
                z[i, j] = row["mean"]
                text[i][j] = f"{row['mean']:+.1f}"
            else:
                text[i][j] = "·"
    lim = max(np.nanmax(np.abs(z)) if np.isfinite(z).any() else 0.5, 0.5)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[[SHORT[v] for v, _ in columns], [regime_mark(v, r) for v, r in columns]],
            y=rows,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color=INK, size=12),
            customdata=custom,
            colorscale=DIVERGING,
            zmin=-lim,
            zmax=lim,
            xgap=2,
            ygap=2,
            colorbar=dict(title="Mean %", thickness=10, outlinewidth=0),
            hovertemplate=(
                "%{y} · %{x}<br>Mean %{z:+.2f}%<br>Median %{customdata[0]:+.2f}%"
                "<br>USD up in %{customdata[1]:.0f}% of quarters"
                "<br>n = %{customdata[2]}<extra></extra>"
            ),
            hoverongaps=False,
        )
    )
    # Separate the four variables' column groups.
    bounds = [j for j in range(1, len(columns)) if columns[j][0] != columns[j - 1][0]]
    for j in bounds:
        fig.add_vline(x=j - 0.5, line=dict(color=SURFACE, width=6))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(plot_bgcolor=NEUTRAL)
    show(fig, height=150 + 42 * len(rows))
    st.caption(
        "Blue = USD strengthened against that currency on average, red = USD weakened. "
        "'·' marks cells with too few quarters. Hover any cell for median, hit rate and n."
    )


def _ranking(stats, series, settings, start, names) -> None:
    st.markdown("**Which currencies win and lose in a regime**")
    c1, c2 = st.columns(2)
    var = c1.selectbox("Variable", DRIVERS, format_func=SHORT.get, key="pr_rank_var")
    avail = [r for r in REGIME_ORDER if ((stats["variable"] == var) & (stats["regime"] == r)).any()]
    reg = c2.radio(
        "Regime",
        avail,
        format_func=lambda r: regime_mark(var, r),
        horizontal=True,
        key=f"pr_rank_reg_{var}",
    )
    d = stats[(stats["variable"] == var) & (stats["regime"] == reg) & stats["shown"]]
    if d.empty:
        st.info("Too few quarters in this regime. Lower the threshold above.")
        return
    d = d.sort_values("mean")  # most negative first = currency strengthened most vs USD

    # Unconditional average over all quarters in the same sample, for comparison.
    uncond = {p: response_returns(s, settings, start).mean() for p, s in series.items()}
    labels = [f"{_ccy(p)}" if p in names else p for p in d["pair"]]
    fig = go.Figure()
    fig.add_bar(
        y=labels,
        x=d["mean"],
        orientation="h",
        name=f"{SHORT[var]} {regime_mark(var, reg)}",
        marker=dict(color=[UP if m > 0 else DOWN for m in d["mean"]]),
        customdata=d[["median", "hit_rate", "n"]].to_numpy(),
        hovertemplate=(
            "%{y}: mean %{x:+.2f}%<br>Median %{customdata[0]:+.2f}%"
            "<br>USD up in %{customdata[1]:.0f}% of quarters<br>n = %{customdata[2]}"
            "<extra></extra>"
        ),
        showlegend=False,
    )
    fig.add_scatter(
        y=labels,
        x=[uncond[p] for p in d["pair"]],
        mode="markers",
        name="All-quarter average",
        marker=dict(symbol="line-ns", size=16, line=dict(color=INK, width=2), color=INK),
        hovertemplate="%{y}: all-quarter average %{x:+.2f}%<extra></extra>",
    )
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_layout(
        xaxis_title="Mean quarterly % move in USD/ccy (+ = USD stronger)",
        hovermode="closest",
        yaxis=dict(autorange="reversed"),
    )
    show(fig, height=120 + 36 * len(labels))
    best, worst = d.iloc[0], d.iloc[-1]
    st.caption(
        f"When {regime_phrase(var, reg)}: **{_ccy(best['pair'])}** has held up best against "
        f"the dollar ({best['mean']:+.2f}% a quarter in USD/{_ccy(best['pair'])}) and "
        f"**{_ccy(worst['pair'])}** worst ({worst['mean']:+.2f}%). The tick marks show each "
        "pair's average over all quarters, so the gap between bar and tick is the regime effect."
    )


def _summary(stats, columns, names, settings) -> None:
    st.markdown("**Strongest and weakest currency vs USD in each regime**")
    defs = definitions(settings)
    rows = []
    for v, r in columns:
        d = stats[
            (stats["variable"] == v)
            & (stats["regime"] == r)
            & stats["pair"].isin(names)
            & stats["shown"]
        ]
        bench = stats[
            (stats["variable"] == v) & (stats["regime"] == r) & (stats["pair"] == BENCHMARK)
        ]
        if d.empty:
            continue
        lo, hi = d.loc[d["mean"].idxmin()], d.loc[d["mean"].idxmax()]
        rows.append(
            {
                "Variable": SHORT[v],
                "Regime": regime_mark(v, r),
                "Definition": defs[v],
                "Quarters": int(d["n"].max()),
                "Strongest ccy": _ccy(lo["pair"]),
                "USD/strongest %": lo["mean"],
                "Weakest ccy": _ccy(hi["pair"]),
                "USD/weakest %": hi["mean"],
                "Pairs with USD up": int((d["mean"] > 0).sum()),
                "Broad USD %": bench["mean"].iloc[0] if not bench.empty else np.nan,
            }
        )
    pct = st.column_config.NumberColumn(format="%+.2f")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "USD/strongest %": pct,
            "USD/weakest %": pct,
            "Broad USD %": pct,
            "Pairs with USD up": st.column_config.NumberColumn(
                help=f"How many of the {len(names)} pairs rose on average (USD stronger)"
            ),
        },
    )

    with st.expander("Full results table"):
        full = stats.assign(
            variable=stats["variable"].map(SHORT),
            regime=[
                regime_mark(v, r) for v, r in zip(stats["variable"], stats["regime"], strict=True)
            ],
        )[["pair", "variable", "regime", "mean", "median", "hit_rate", "n"]]
        st.dataframe(
            full,
            hide_index=True,
            width="stretch",
            column_config={
                "mean": st.column_config.NumberColumn("Mean %", format="%+.2f"),
                "median": st.column_config.NumberColumn("Median %", format="%+.2f"),
                "hit_rate": st.column_config.NumberColumn("USD up (% of qtrs)", format="%.0f"),
                "n": "Quarters",
            },
        )
        st.download_button(
            "Download CSV", full.to_csv(index=False).encode(), "pair_regimes.csv", "text/csv"
        )

    st.caption(
        "Strongest currency = the pair that fell most on average (USD weakened most against "
        "it); weakest = the pair that rose most. USDCNH uses onshore CNY. These are historical "
        "averages, not forecasts, and regimes overlap, so effects are not additive."
    )
