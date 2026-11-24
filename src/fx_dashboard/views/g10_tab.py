"""Tab 6: the US against one other G10 economy. Each side's macro regime, the gap between
them, and what the USD spot did in every combination (e.g. Fed hiking while the BoJ holds)."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fx_dashboard import yields2y
from fx_dashboard.analysis import (
    REGIME_ORDER,
    classify,
    fit_sensitivity,
    gap_betas,
    gap_implied_spot,
    joint_regime_stats,
    pct,
    policy_cycle,
    quarterly_returns,
    regime_stats,
    to_quarterly,
    weekly_spot_and_gap,
)
from fx_dashboard.countries import COUNTRIES, Country
from fx_dashboard.series import MACRO_VARS, US_3M, US_10Y3M
from fx_dashboard.theme import (
    AXIS,
    DIVERGING,
    DOWN,
    INK,
    MUTED,
    NEUTRAL,
    REGIME_COLORS,
    REGIME_SYMBOLS,
    SERIES,
    SURFACE,
    UP,
    show,
)

COMPARE = {
    "FEDFUNDS": "Policy rate",
    "Y2": "2Y govt yield",
    "DGS2": "Short rate",
    "T10Y2Y": "Yield curve",
    "CPI": "Inflation",
    "GDP": "Growth",
}
# Compared in levels; CPI and GDP as y/y % change.
RATES = {"FEDFUNDS", "Y2", "DGS2", "T10Y2Y"}
# 2Y government yields from the bundled Bundesbank / MoF Japan / Bank of England files.
Y2 = yields2y.load()
NAMES = {
    "FEDFUNDS": {"Up": "Hiking", "Flat": "On hold", "Down": "Cutting"},
    "Y2": {"Up": "Rising", "Flat": "Steady", "Down": "Falling"},
    "DGS2": {"Up": "Rising", "Flat": "Steady", "Down": "Falling"},
    "T10Y2Y": {"Up": "Steepening", "Flat": "Steady", "Down": "Flattening"},
    "CPI": {"Up": "Rising", "Flat": "Steady", "Down": "Falling"},
    "GDP": {"Up": "Accelerating", "Flat": "Steady", "Down": "Slowing"},
}
GAP_NAMES = {"Up": "Gap rising", "Flat": "Gap steady", "Down": "Gap falling"}
CENTRAL_BANK = {
    "AU": "RBA",
    "CA": "BoC",
    "FR": "ECB",
    "DE": "ECB",
    "IT": "ECB",
    "JP": "BoJ",
    "NZ": "RBNZ",
    "NO": "Norges Bank",
    "SE": "Riksbank",
    "CH": "SNB",
    "GB": "BoE",
}
MEASURES = {"Average quarterly %": "mean", "Annualised %": "ann"}
US_COLOR, FX_COLOR = SERIES[0], SERIES[1]
POLICY_BAND = 0.01  # official rates move in steps of 5bp or more; this only absorbs float noise
EFFECTIVE_FF_BAND = 0.10  # if the Fed target fails to load, the effective rate drifts between moves


@dataclass(frozen=True)
class Side:
    """One economy's version of a variable."""

    raw: pd.Series
    label: str  # e.g. "Fed target rate", "BoJ policy rate"
    who: str  # e.g. "Fed", "BoJ", "US", "Japan"
    freq: str  # "M" | "Q": publication frequency of an index (for y/y)
    band: float | None = None  # policy only: deadband for a "move"


def _sides(
    key: str,
    macro: dict[str, pd.Series],
    fed_target: pd.Series | None,
    us_extra: dict[str, pd.Series],
    country: Country,
    fdata: dict,
) -> tuple[Side | None, Side | None, str]:
    """US and foreign series for one comparison, plus a note on any mismatch."""
    if key == "Y2":
        us = Side(macro["DGS2"], "2Y Treasury yield", "US", "M")
        fx_raw = Y2.get(country.code)
        label = yields2y.LABELS.get(country.code, "")
        fx = Side(fx_raw, label, country.name, "M") if fx_raw is not None else None
        return us, fx, ""
    fvar = country.vars[key]
    proxy = fvar.short == "3M rate"  # the foreign short end is a 3M rate, not a 2Y yield
    note = ""
    if key == "FEDFUNDS":
        if fed_target is not None:
            us = Side(fed_target, "Fed target rate (upper bound)", "Fed", "M", POLICY_BAND)
        else:
            us = Side(macro["FEDFUNDS"], "Fed funds (effective)", "Fed", "M", EFFECTIVE_FF_BAND)
        cb = CENTRAL_BANK[country.code]
        fx_raw = fdata["macro"].get(key)
        fx = Side(fx_raw, country.policy_name, cb, "M", POLICY_BAND) if fx_raw is not None else None
        return us, fx, note
    us_id = {"DGS2": US_3M, "T10Y2Y": US_10Y3M}.get(key) if proxy else None
    if us_id and us_id in us_extra:
        us_raw = us_extra[us_id]
        us_label = "3M Treasury bill" if key == "DGS2" else "10Y Treasury − 3M bill"
    else:
        us_raw, us_label = macro[key], MACRO_VARS[key].label
        if us_id:
            note = (
                f"The US {'3M bill' if key == 'DGS2' else '10Y−3M spread'} failed to load, so "
                f"the US {MACRO_VARS[key].short} is compared with {country.name}'s "
                f"{fvar.short}: not like for like."
            )
    us = Side(us_raw, us_label, "US", MACRO_VARS[key].freq)
    fx_raw = fdata["macro"].get(key)
    fx = Side(fx_raw, fvar.label, country.name, fvar.freq) if fx_raw is not None else None
    return us, fx, note


def _quarterly(key: str, side: Side) -> pd.Series:
    """Quarter-end level for rates; y/y % change for CPI and GDP."""
    q = to_quarterly(side.raw)
    return q if key in RATES else pct(q, 4)


def _monthly(key: str, side: Side) -> pd.Series:
    if key in RATES:
        return side.raw.resample("ME").last().dropna()
    s = side.raw.resample("QE" if side.freq == "Q" else "ME").last().dropna()
    return pct(s, 4 if side.freq == "Q" else 12).dropna()


def _regime(key: str, q: pd.Series, band: float, cycle: bool, max_pause: int) -> pd.Series:
    moves = classify(q.diff(), band)
    return policy_cycle(moves, max_pause) if key == "FEDFUNDS" and cycle else moves


def _mark(key: str, reg) -> str:
    return f"{REGIME_SYMBOLS[reg]} {NAMES[key][reg]}" if isinstance(reg, str) else "n/a"


def _unit(key: str) -> str:
    return {"T10Y2Y": "pp"}.get(key, "%") if key in RATES else "% y/y"


def render(
    macro: dict[str, pd.Series],
    pairs: dict[str, pd.Series],
    us_extra: dict[str, pd.Series],
    start,
    fed_target: pd.Series | None,
    load_country: Callable[[str], dict],
) -> None:
    st.subheader("G10 vs US: two economies' regimes and the dollar")
    st.caption(
        "Pick an economy and a variable. Each side gets its own regime each quarter (e.g. Fed "
        "hiking, BoJ on hold), and the table shows what USD/that currency did in every "
        f"combination. Quarterly, from {pd.Timestamp(start):%b %Y} (sidebar). Pairs are "
        "foreign currency per USD: positive = USD stronger."
    )
    c_ctry, c_var = st.columns([3, 2])
    code = (
        c_ctry.segmented_control(
            "Economy",
            list(COUNTRIES),
            format_func=lambda c: COUNTRIES[c].name,
            default="JP",
            key="g10_country",
        )
        or "JP"
    )
    key = (
        c_var.segmented_control(
            "Compare", list(COMPARE), format_func=COMPARE.get, default="FEDFUNDS", key="g10_var"
        )
        or "FEDFUNDS"
    )
    country = COUNTRIES[code]
    pair = f"USD{country.currency}"

    c_resp, c_meas = st.columns(2)
    response = c_resp.radio(
        f"{pair} response measured",
        ["Same quarter", "Next quarter"],
        horizontal=True,
        key="g10_response",
        help="'Next quarter' compares this quarter's regimes with the following quarter's "
        "move, to check whether regimes lead the currency.",
    )
    measure_label = c_meas.radio("Measure", list(MEASURES), horizontal=True, key="g10_measure")
    m = MEASURES[measure_label]
    with st.expander("Regime thresholds"):
        a, b, c, d = st.columns(4)
        cycle = a.toggle(
            "Policy: pauses inside a cycle count",
            value=True,
            key="g10_cycle",
            help="A quarter with no move between two hikes (or two cuts) stays part of the "
            "hiking (cutting) cycle.",
        )
        max_pause = b.number_input(
            "Longest pause inside a cycle (quarters)",
            0,
            8,
            4,
            1,
            key="g10_max_pause",
            disabled=not cycle,
        )
        band_rate = (
            c.number_input(
                "Yields, short rate & curve: steady band (bps)",
                0,
                100,
                10,
                5,
                key="g10_band_rate",
            )
            / 100
        )
        band_idx = d.number_input(
            "Inflation & growth: steady band (pp of y/y)",
            0.0,
            2.0,
            0.1,
            0.05,
            key="g10_band_idx",
        )

    fdata = load_country(code)
    px = pairs.get(pair)
    us, fx, note = _sides(key, macro, fed_target, us_extra, country, fdata)
    if fx is None and key == "Y2":
        names = [COUNTRIES[c].name for c in Y2]
        names = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names)
        st.info(
            f"2Y government yields (from the Bundesbank, MoF Japan and Bank of England files) "
            f"cover {names}. Pick one of those, or choose **Short rate** to compare "
            f"{country.name}'s 3-month rate with the US 3-month bill."
        )
        return
    if fx is None:
        st.warning(
            f"{country.name}'s {COMPARE[key].lower()} failed to load: "
            f"{fdata['errors'].get(key, 'unknown error')}. Press Refresh data to retry."
        )
        return
    if px is None:
        st.warning(f"{pair} failed to load. Press Refresh data to retry.")
        return
    if note:
        st.info(note)

    start_ts = pd.Timestamp(start)
    band = band_rate if key in ("Y2", "DGS2", "T10Y2Y") else band_idx
    us_q, fx_q = _quarterly(key, us), _quarterly(key, fx)
    us_reg = _regime(key, us_q, us.band if us.band is not None else band, cycle, max_pause)
    fx_reg = _regime(key, fx_q, fx.band if fx.band is not None else band, cycle, max_pause)
    gap_q = (us_q - fx_q).dropna()
    gap_reg = classify(gap_q.diff(), POLICY_BAND if key == "FEDFUNDS" else band)
    ret = quarterly_returns(px)
    if response == "Next quarter":
        ret = ret.shift(-1)

    def cut(s: pd.Series) -> pd.Series:
        return s[s.index >= start_ts]

    us_reg, fx_reg, gap_q, gap_reg, ret = map(cut, (us_reg, fx_reg, gap_q, gap_reg, ret))
    js = joint_regime_stats(ret, us_reg, fx_reg)
    if js.empty:
        st.warning("No overlapping quarters in the selected sample.")
        return

    _snapshot(macro, fed_target, us_extra, country, fdata)
    if key == "Y2":
        _y2_differential(us, fx, px, pair, start_ts, country)
        st.markdown(f"**Quarterly regimes: 2Y Treasury vs {fx.label}**")
    _tiles(key, us, fx, us_reg, fx_reg, gap_q, js, pair, m, measure_label)
    _timeline(key, us, fx, px, us_reg, fx_reg, gap_reg, start_ts, country, pair)
    left, right = st.columns([3, 2])
    with left:
        _matrix(key, js, m, measure_label, us, fx, us_reg, fx_reg, pair)
    with right:
        _gap_bars(key, ret, gap_reg, m, measure_label, us, fx, pair)
    _scatter(key, gap_q, ret, us, fx, pair, response)
    _table(key, us_q, fx_q, gap_q, us_reg, fx_reg, gap_reg, ret, us, fx, pair, start_ts)

    euro = (
        " France, Germany and Italy share the ECB rate and the euro, so their policy-rate "
        "results are identical; their other variables differ."
        if country.currency == "EUR"
        else ""
    )
    st.caption(
        f"Regimes: policy rates are dated from each central bank's official rate (hiking / "
        f"cutting = a move in the quarter, or a pause of up to {max_pause} quarters inside a "
        "cycle when that option is on). Other variables: quarterly change in the level (short "
        "rate, curve) or in the y/y rate (inflation, growth), with changes inside the steady "
        "band labelled steady. Averages describe history, not forecasts; cells with only a few "
        f"quarters can be dominated by one episode.{euro}"
    )


def _fmt(key: str, v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    return f"{v:+.2f}pp" if key == "T10Y2Y" else f"{v:.2f}%"


def _snapshot(macro, fed_target, us_extra, country: Country, fdata: dict) -> None:
    """Latest value of every variable for both economies and the gap between them."""
    rows = []
    for k, label in COMPARE.items():
        us, fx, _ = _sides(k, macro, fed_target, us_extra, country, fdata)
        if fx is None:
            continue
        us_m, fx_m = _monthly(k, us), _monthly(k, fx)
        gap = (us_m - fx_m).dropna()
        if gap.empty:
            continue
        ago = gap.asof(gap.index[-1] - pd.DateOffset(years=1))
        rows.append(
            {
                "Variable": f"{label} ({_unit(k)})",
                # both sides on the gap's date, so the columns subtract to the gap
                "United States": us_m[gap.index[-1]],
                country.name: fx_m[gap.index[-1]],
                f"Gap (US − {country.code})": gap.iloc[-1],
                "Gap a year earlier": ago,
                "Gap change": gap.iloc[-1] - ago,
                "As of": f"{gap.index[-1]:%b %Y}",
                "US measure": us.label,
                f"{country.name} measure": fx.label,
            }
        )
    st.markdown(f"**United States vs {country.name} today**")
    num = st.column_config.NumberColumn(format="%.2f")
    chg = st.column_config.NumberColumn(format="%+.2f")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "United States": num,
            country.name: num,
            f"Gap (US − {country.code})": chg,
            "Gap a year earlier": chg,
            "Gap change": chg,
        },
    )


BETA_WINDOWS = {"Last 1 year": 52, "Last 3 years": 156, "Whole sample": None}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _y2_differential(us: Side, fx: Side, px: pd.Series, pair: str, start_ts, country) -> None:
    """Daily US 2Y Treasury minus foreign 2Y government yield, and how the spot tracks it."""
    st.markdown(f"**2Y yield differential and {pair}** · US 2Y Treasury − {fx.label}, daily")
    daily = pd.concat({"us": us.raw, "fx": fx.raw}, axis=1).dropna()
    daily = daily[daily.index >= start_ts]
    if len(daily) < 60:
        st.info("Too few days where both 2Y yields are available in the selected sample.")
        return
    gap = (daily["us"] - daily["fx"]).rename("gap")
    last = gap.index[-1]

    def change_bp(months: int) -> float:
        base = gap.asof(last - pd.DateOffset(months=months))
        return (gap.iloc[-1] - base) * 100

    weekly = weekly_spot_and_gap(px[px.index >= start_ts], gap)
    cols = st.columns(5)
    cols[0].metric("US 2Y Treasury", f"{daily['us'].iloc[-1]:.2f}%")
    cols[1].metric(fx.label, f"{daily['fx'].iloc[-1]:.2f}%")
    cols[2].metric(
        f"Differential (US − {country.code})",
        f"{gap.iloc[-1]:+.2f}pp",
        delta=f"{change_bp(1):+.0f}bp 1M · {change_bp(3):+.0f}bp 3M · {change_bp(12):+.0f}bp 1Y",
        delta_color="off",
        delta_arrow="off",
        help=f"As of {last:%d %b %Y}, the latest day both yields are available.",
    )
    pctile = (gap <= gap.iloc[-1]).mean() * 100
    cols[3].metric(
        "Percentile of history",
        _ordinal(round(pctile)),
        help=f"Share of days since {gap.index[0]:%b %Y} with a differential at or below today's.",
    )
    if len(weekly) >= 52:
        implied, _, r2 = gap_implied_spot(weekly, 52)
        spot_now = weekly["spot"].iloc[-1]
        cols[4].metric(
            f"{pair} implied by the gap",
            f"{implied:.4g}",
            delta=f"spot {spot_now:.4g}, {(spot_now / implied - 1) * 100:+.1f}% vs implied",
            delta_color="off",
            delta_arrow="off",
            help=f"Straight-line fit of weekly {pair} on the 2Y differential over the last 52 "
            f"weeks (R² {r2:.2f}), evaluated at today's differential. A rough gauge of how far "
            "spot sits from what the rate gap alone suggests, not a forecast.",
        )

    gap_w = weekly["gap"]
    chg = pd.DataFrame({"y": pct(weekly["spot"]), "x": gap_w.diff()}).dropna()
    corr = chg["y"].rolling(52).corr(chg["x"]).dropna()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": True}], [{}]],
    )
    fig.add_scatter(
        x=weekly.index,
        y=weekly["spot"],
        mode="lines",
        name=f"{pair} (left)",
        line=dict(color=INK, width=2),
        hovertemplate="%{x|%d %b %Y}<br>" + pair + " %{y:.4g}<extra></extra>",
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_scatter(
        x=gap_w.index,
        y=gap_w.values,
        mode="lines",
        name=f"US − {country.name} 2Y, pp (right)",
        line=dict(color=US_COLOR, width=2),
        hovertemplate="%{x|%d %b %Y}<br>Differential %{y:+.2f}pp<extra></extra>",
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_scatter(
        x=corr.index,
        y=corr.values,
        mode="lines",
        name="52-week correlation of weekly changes",
        line=dict(color=FX_COLOR, width=2),
        fill="tozeroy",
        fillcolor="rgba(217,89,38,0.2)",
        hovertemplate="%{x|%d %b %Y}<br>Correlation %{y:.2f}<extra></extra>",
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=2, col=1)
    fig.update_yaxes(title_text=pair, row=1, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text="US − foreign 2Y, pp", row=1, col=1, secondary_y=True, showgrid=False
    )
    fig.update_yaxes(title_text="correlation", range=[-1, 1], row=2, col=1)
    fig.update_layout(hovermode="x unified")
    show(fig, height=560)

    betas = gap_betas(weekly, BETA_WINDOWS)
    if not betas.empty:
        table = pd.DataFrame(
            {
                "Window": betas["window"],
                "From": betas["from"].dt.strftime("%b %Y"),
                f"{pair} % per 10bp wider gap": betas["beta_10bp"],
                "t-stat": betas["t"],
                "Correlation": betas["corr"],
                "R²": betas["r2"],
                "Weeks": betas["weeks"],
            }
        )
        two = st.column_config.NumberColumn(format="%.2f")
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                f"{pair} % per 10bp wider gap": st.column_config.NumberColumn(format="%+.2f"),
                "t-stat": st.column_config.NumberColumn(format="%.1f"),
                "Correlation": two,
                "R²": two,
            },
        )
        b = betas.iloc[0]
        verdict = "a reliable link" if abs(b["t"]) >= 2 else "too noisy to call a reliable link"
        st.caption(
            f"{b['window']}: each 10bp the US–{country.name} 2Y differential widened went with "
            f"{pair} {b['beta_10bp']:+.2f}% the same week (correlation {b['corr']:.2f}, "
            f"t = {b['t']:.1f}): {verdict}. Positive = the textbook case, a wider US yield "
            "advantage goes with a stronger dollar. Weekly changes, Friday to Friday."
        )
    st.caption(
        f"US 2Y: FRED DGS2. {country.name} 2Y: {yields2y.SOURCES[country.code]}, from your "
        f"downloaded file (last observation {fx.raw.index[-1]:%d %b %Y}). To extend it, "
        "download fresh copies of the three files and run "
        '`python -m fx_dashboard.yields2y "<folder>"`.'
    )
    out = daily.assign(gap=gap).join(px.rename(pair), how="left")
    out.columns = ["US 2Y %", f"{country.code} 2Y %", "Differential pp", pair]
    st.download_button(
        "Download daily differential CSV",
        out.sort_index(ascending=False).to_csv().encode(),
        file_name=f"us_vs_{country.code.lower()}_2y_differential.csv",
        mime="text/csv",
        key="g10_y2_csv",
    )


def _tiles(key, us: Side, fx: Side, us_reg, fx_reg, gap_q, js, pair, m, measure_label) -> None:
    both = pd.concat({"us": us_reg, "fx": fx_reg}, axis=1).dropna()
    cols = st.columns(4)
    if both.empty:
        return
    q = both.index[-1]
    us_now, fx_now = both.iloc[-1]["us"], both.iloc[-1]["fx"]
    cols[0].metric(f"{us.who} now", _mark(key, us_now), help=f"Quarter {q.to_period('Q')}")
    cols[1].metric(f"{fx.who} now", _mark(key, fx_now), help=f"Quarter {q.to_period('Q')}")
    g = gap_q.dropna()
    ago = g.asof(g.index[-1] - pd.DateOffset(years=1))
    unit = "pp"
    cols[2].metric(
        f"Gap: {us.who} − {fx.who}",
        f"{g.iloc[-1]:+.2f}{unit}",
        delta=f"{g.iloc[-1] - ago:+.2f}{unit} vs a year earlier" if np.isfinite(ago) else None,
        delta_color="off",
        delta_arrow="off",
        help=f"Quarter-end {g.index[-1].to_period('Q')}",
    )
    cell = js[(js["a"] == us_now) & (js["b"] == fx_now)]
    if cell.empty:
        cols[3].metric(f"{pair} in this mix", "no history")
    else:
        r = cell.iloc[0]
        cols[3].metric(
            f"{pair} in this mix",
            f"{r[m]:+.2f}%" + (" a year" if m == "ann" else " a qtr"),
            delta=f"USD up in {r['hit_rate']:.0f}% of {int(r['n'])} qtrs",
            delta_color="off",
            delta_arrow="off",
            help=f"{measure_label} of {pair} in past quarters when the {us.who} was "
            f"{NAMES[key][us_now].lower()} and the {fx.who} was {NAMES[key][fx_now].lower()}.",
        )


def _regime_z(reg: pd.Series) -> pd.Series:
    return reg.map({"Up": 1, "Flat": 0, "Down": -1}).astype(float)


def _timeline(key, us: Side, fx: Side, px, us_reg, fx_reg, gap_reg, start_ts, country, pair):
    st.markdown(f"**{pair}, both economies' {COMPARE[key].lower()} and their regimes**")
    spot = px[px.index >= start_ts].resample("W-FRI").last().dropna()
    us_m, fx_m = (s[s.index >= start_ts] for s in (_monthly(key, us), _monthly(key, fx)))
    gap = (us_m - fx_m).dropna()
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.36, 0.3, 0.18, 0.16],
        vertical_spacing=0.04,
    )
    fig.add_scatter(
        x=spot.index,
        y=spot.values,
        mode="lines",
        name=pair,
        line=dict(color=INK, width=2),
        hovertemplate="%{x|%d %b %Y}<br>" + pair + " %{y:.4f}<extra></extra>",
        showlegend=False,
        row=1,
        col=1,
    )
    unit = _unit(key)
    for s, side, color in ((us_m, us, US_COLOR), (fx_m, fx, FX_COLOR)):
        fig.add_scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=f"{side.who}: {side.label}",
            line=dict(color=color, width=2, shape="hv" if key == "FEDFUNDS" else "linear"),
            hovertemplate="%{x|%b %Y}<br>" + side.who + " %{y:.2f}" + unit + "<extra></extra>",
            row=2,
            col=1,
        )
    fig.add_scatter(
        x=gap.index,
        y=gap.values,
        mode="lines",
        name=f"Gap: {us.who} − {fx.who}",
        line=dict(color=MUTED, width=2),
        fill="tozeroy",
        fillcolor="rgba(137,135,129,0.25)",
        hovertemplate="%{x|%b %Y}<br>Gap %{y:+.2f}pp<extra></extra>",
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=3, col=1)

    strip = pd.concat({us.who: us_reg, fx.who: fx_reg, "Gap": gap_reg}, axis=1).dropna(how="all")
    x = [i.to_period("Q").start_time + pd.Timedelta(days=45) for i in strip.index]
    z = np.vstack([_regime_z(strip[c]).to_numpy() for c in strip.columns])
    names = [NAMES[key], NAMES[key], GAP_NAMES]
    text = [
        [names[i].get(v, "") if isinstance(v, str) else "" for v in strip[c]]
        for i, c in enumerate(strip.columns)
    ]
    fig.add_heatmap(
        x=x,
        y=list(strip.columns),
        z=z,
        text=text,
        zmin=-1.5,
        zmax=1.5,
        colorscale=[
            [0.0, DOWN],
            [0.333, DOWN],
            [0.333, MUTED],
            [0.667, MUTED],
            [0.667, UP],
            [1.0, UP],
        ],
        showscale=False,
        xgap=1,
        ygap=3,
        hovertemplate="%{x|%Y Q%q}<br>%{y}: %{text}<extra></extra>",
        hoverongaps=False,
        row=4,
        col=1,
    )
    fig.update_yaxes(title_text=pair, row=1, col=1)
    fig.update_yaxes(title_text=unit, row=2, col=1)
    fig.update_yaxes(title_text="gap, pp", row=3, col=1)
    fig.update_yaxes(autorange="reversed", showgrid=False, row=4, col=1)
    fig.update_layout(hovermode="x unified", plot_bgcolor=SURFACE)
    show(fig, height=720)
    st.caption(
        "Bottom strip, one cell per quarter: blue = "
        f"{NAMES[key]['Up'].lower()} / gap rising, red = {NAMES[key]['Down'].lower()} / gap "
        f"falling, grey = {NAMES[key]['Flat'].lower()}. The gap is {us.who} minus {fx.who}: "
        "a rising gap means the US side is moving up relative to the other economy."
    )


def _matrix(key, js, m, measure_label, us: Side, fx: Side, us_reg, fx_reg, pair) -> None:
    st.markdown(f"**{pair} in each combination of regimes** · {measure_label.lower()}")
    z = np.full((3, 3), np.nan)
    custom = np.full((3, 3, 2), np.nan)
    text = [["" for _ in range(3)] for _ in range(3)]
    for _, r in js.iterrows():
        i, j = REGIME_ORDER.index(r["a"]), REGIME_ORDER.index(r["b"])
        z[i, j] = r[m]
        custom[i, j] = [r["hit_rate"], r["n"]]
        text[i][j] = f"{r[m]:+.2f}%<br>{int(r['n'])} qtrs"
    xs = [f"{fx.who} {NAMES[key][g].lower()}" for g in REGIME_ORDER]
    ys = [f"{us.who} {NAMES[key][g].lower()}" for g in REGIME_ORDER]
    lim = max(float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0, 0.5)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=xs,
            y=ys,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color=INK, size=14),
            customdata=custom,
            colorscale=DIVERGING,
            zmin=-lim,
            zmax=lim,
            xgap=3,
            ygap=3,
            colorbar=dict(title="%", thickness=10, outlinewidth=0),
            hovertemplate="%{y} · %{x}<br>" + pair + " %{z:+.2f}%"
            "<br>USD up in %{customdata[0]:.0f}% of %{customdata[1]} quarters<extra></extra>",
            hoverongaps=False,
        )
    )
    both = pd.concat({"us": us_reg, "fx": fx_reg}, axis=1).dropna()
    if not both.empty:
        i = REGIME_ORDER.index(both.iloc[-1]["us"])
        j = REGIME_ORDER.index(both.iloc[-1]["fx"])
        fig.add_shape(
            type="rect",
            x0=j - 0.5,
            x1=j + 0.5,
            y0=i - 0.5,
            y1=i + 0.5,
            line=dict(color=INK, width=3),
        )
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(plot_bgcolor=NEUTRAL)
    show(fig, height=380)
    st.caption(
        f"Blue = USD gained against {pair[3:]} on average, red = USD lost. The outlined cell "
        "is the latest quarter's combination. Empty cells never happened in the sample."
    )


def _gap_bars(key, ret, gap_reg, m, measure_label, us: Side, fx: Side, pair) -> None:
    st.markdown(f"**{pair} when the gap ({us.who} − {fx.who}) is rising or falling**")
    rs = regime_stats(ret, gap_reg.to_frame("GAP")).set_index("regime")
    regs = [r for r in REGIME_ORDER if r in rs.index]
    if not regs:
        st.info("No gap regimes in the sample.")
        return
    fig = go.Figure(
        go.Bar(
            x=[GAP_NAMES[r] for r in regs],
            y=[rs.loc[r, m] for r in regs],
            marker_color=[REGIME_COLORS[r] for r in regs],
            text=[f"{rs.loc[r, m]:+.2f}%" for r in regs],
            textposition="outside",
            customdata=[[rs.loc[r, "hit_rate"], rs.loc[r, "n"]] for r in regs],
            hovertemplate="%{x}<br>" + pair + " %{y:+.2f}%"
            "<br>USD up in %{customdata[0]:.0f}% of %{customdata[1]} quarters<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_layout(yaxis_title=f"{measure_label} (+ = USD stronger)", showlegend=False)
    show(fig, height=380)
    st.caption(
        "Gap rising = the US side moved up relative to the other economy over the quarter "
        "(e.g. the Fed hiked or the other bank cut)."
    )


def _scatter(key, gap_q, ret, us: Side, fx: Side, pair, response) -> None:
    st.markdown(f"**Does the gap move {pair}?** · each dot is one quarter")
    df = pd.concat({"x": gap_q.diff(), "y": ret}, axis=1).dropna()
    if len(df) < 8:
        st.info("Too few quarters for a regression.")
        return
    sens = fit_sensitivity(df["y"], df[["x"]], min_obs=8)
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    xs = np.linspace(df["x"].min(), df["x"].max(), 50)
    fig = go.Figure()
    fig.add_scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        name="Quarter",
        marker=dict(color=US_COLOR, size=8, opacity=0.75, line=dict(color=SURFACE, width=1)),
        customdata=[str(i.to_period("Q")) for i in df.index],
        hovertemplate="%{customdata}<br>Gap change %{x:+.2f}pp<br>"
        + pair
        + " %{y:+.2f}%<extra></extra>",
    )
    fig.add_scatter(
        x=[df["x"].iloc[-1]],
        y=[df["y"].iloc[-1]],
        mode="markers+text",
        name="Latest",
        text=[str(df.index[-1].to_period("Q"))],
        textposition="top center",
        marker=dict(color=INK, size=12, symbol="diamond"),
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=xs,
        y=intercept + slope * xs,
        mode="lines",
        name="Fitted line",
        line=dict(color=FX_COLOR, width=2),
        hoverinfo="skip",
    )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    when = "the same quarter" if response == "Same quarter" else "the next quarter"
    fig.update_layout(
        xaxis_title=f"Quarterly change in the gap, {us.who} − {fx.who} (pp)",
        yaxis_title=f"{pair} % change, {when} (+ = USD stronger)",
        hovermode="closest",
    )
    show(fig, height=420)
    if sens is not None:
        t = float(sens.tstat["x"])
        verdict = "a reliable link" if abs(t) >= 2 else "too noisy to call a reliable link"
        st.caption(
            f"Each 1pp the gap rose went with {pair} {float(sens.beta['x']):+.2f}% in {when} "
            f"(t = {t:.1f}, R² = {sens.r2:.2f}, {sens.nobs} quarters): {verdict}. A positive "
            "slope is the textbook case: the US side rising relative to the other economy "
            "lifts the dollar."
        )


def _table(key, us_q, fx_q, gap_q, us_reg, fx_reg, gap_reg, ret, us, fx, pair, start_ts):
    with st.expander("Quarterly data"):
        unit = _unit(key)
        table = pd.DataFrame(
            {
                f"{us.who} ({unit})": us_q,
                f"{fx.who} ({unit})": fx_q,
                "Gap (pp)": gap_q,
                f"{us.who} regime": us_reg.map(NAMES[key]),
                f"{fx.who} regime": fx_reg.map(NAMES[key]),
                "Gap regime": gap_reg.map(GAP_NAMES),
                f"{pair} % (response)": ret,
            }
        )
        table = table[table.index >= start_ts].dropna(how="all").sort_index(ascending=False)
        table.index = table.index.to_period("Q").astype(str)
        table.index.name = "Quarter"
        st.dataframe(table, width="stretch")
        st.download_button(
            "Download CSV",
            table.to_csv().encode(),
            file_name=f"us_vs_{fx.who.lower().replace(' ', '_')}_{key.lower()}.csv",
            mime="text/csv",
            key="g10_csv",
        )
