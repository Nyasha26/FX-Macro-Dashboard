"""Chart styling: one Plotly template plus fixed colour assignments.

Colours follow the entity, never its rank: each macro driver and each regime keeps
its colour whatever is toggled on or off.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Dark theme on a black page. Series steps are the dark-surface set, validated on #000000.
SURFACE = "#000000"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
NEUTRAL = "#383835"  # diverging midpoint / empty heatmap cells
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
DRIVER_COLORS = {
    "FEDFUNDS": SERIES[0],
    "DGS2": SERIES[1],
    "CPI": SERIES[2],
    "GDP": SERIES[3],
    "T10Y2Y": SERIES[4],
}

# Diverging pair: blue = up, red = down, grey = flat / zero.
UP, DOWN = "#3987e5", "#e66767"
REGIME_COLORS = {"Up": UP, "Flat": MUTED, "Down": DOWN}
REGIME_MARKS = {"Up": "▲ Up", "Flat": "■ Flat", "Down": "▼ Down"}
REGIME_SYMBOLS = {"Up": "▲", "Flat": "■", "Down": "▼"}
# Variables whose regimes have their own names; the rest use Up / Flat / Down.
REGIME_NAMES = {"FEDFUNDS": {"Up": "Hiking", "Flat": "On hold", "Down": "Cutting"}}
DIVERGING = [[0.0, DOWN], [0.5, NEUTRAL], [1.0, UP]]


def regime_name(var: str, regime: str) -> str:
    return REGIME_NAMES.get(var, {}).get(regime, regime)


def regime_mark(var: str, regime: str) -> str:
    """Symbol plus name, e.g. '▲ Hiking' for fed funds or '▼ Down' for the 2Y yield."""
    return f"{REGIME_SYMBOLS[regime]} {regime_name(var, regime)}"


_axis = dict(
    gridcolor=GRID,
    gridwidth=1,
    linecolor=AXIS,
    zerolinecolor=AXIS,
    zerolinewidth=1,
    tickfont=dict(color=INK_2),
    title=dict(font=dict(color=INK_2)),
    ticks="",
    automargin=True,
)

pio.templates["fxdash"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=13, color=INK),
        colorway=SERIES,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis=_axis,
        yaxis=_axis,
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left", title_text=""),
        hoverlabel=dict(font=dict(family=FONT)),
        barcornerradius=4,
        bargap=0.3,
        bargroupgap=0.08,
    ),
    data=dict(scatter=[go.Scatter(line=dict(width=2))]),
)
pio.templates.default = "plotly_white+fxdash"


def show(fig: go.Figure, height: int | None = None) -> None:
    """Render with our template rather than Streamlit's theme."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


def line(s, color: str, y_title: str = "", name: str = "", zero_line: bool = False) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=name or str(s.name),
            line=dict(color=color, width=2),
            hovertemplate="%{x|%b %Y}<br>%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(yaxis_title=y_title, showlegend=False, hovermode="x")
    if zero_line:
        fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    return fig
