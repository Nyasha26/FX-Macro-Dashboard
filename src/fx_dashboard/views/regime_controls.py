"""Regime settings shared by the dollar-index, pair-regime and Fed-cycle tabs."""

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from fx_dashboard.analysis import (
    DRIVERS,
    classify,
    fed_regime,
    policy_cycle,
    quarterly_returns,
    regime_signals,
)
from fx_dashboard.series import MACRO_VARS
from fx_dashboard.theme import regime_name

SHORT = {k: MACRO_VARS[k].short for k in DRIVERS}
FED_DEFINITIONS = {
    "cycle": "Policy cycle (pauses inside a cycle count)",
    "quarter": "Each quarter's decision",
}
# Used only if the Fed target series fails to load and the effective rate stands in.
EFFECTIVE_FF_DEADBAND = 0.10


@dataclass(frozen=True)
class RegimeSettings:
    basis: str  # "momentum" | "direction"
    response: str  # "Same quarter" | "Next quarter"
    deadband: dict[str, float]
    fed_cycle: bool = True
    max_pause: int = 4


def definitions(settings: RegimeSettings) -> dict[str, str]:
    if settings.basis == "momentum":
        cpi, gdp = "YoY inflation rising / falling", "YoY real growth accelerating / slowing"
    else:
        cpi, gdp = "CPI index up / down on the quarter", "Real GDP up / down on the quarter"
    if settings.fed_cycle:
        fed = (
            "Hiking / Cutting: the Fed raised / cut its target in the quarter, or is pausing "
            f"for up to {settings.max_pause} quarters between moves in the same direction; "
            "On hold otherwise"
        )
    else:
        fed = "Hiking / Cutting: the Fed raised / cut its target in the quarter; On hold otherwise"
    return {
        "FEDFUNDS": fed,
        "DGS2": "2Y yield up / down on the quarter",
        "CPI": cpi,
        "GDP": gdp,
        "T10Y2Y": "Curve steepening (spread up) / flattening (spread down) on the quarter",
    }


def regime_phrase(var: str, regime: str) -> str:
    """Readable clause for captions: 'the Fed is hiking', 'the 2Y yield is up'."""
    if var == "FEDFUNDS":
        return f"the Fed is {regime_name(var, regime).lower()}"
    return f"{SHORT[var]} is {regime.lower()}"


def _fed_controls(prefix: str, col) -> bool:
    label = col.radio(
        "Fed regime definition",
        list(FED_DEFINITIONS.values()),
        horizontal=True,
        key=f"{prefix}_fed_def",
        help="Hiking and cutting are dated from the Fed's policy target (the top of the "
        "range since Dec 2008), not the monthly-average effective rate. 'Policy cycle' keeps "
        "a pause between two hikes (or two cuts) inside the cycle, e.g. 2016 between the "
        "Dec-2015 and Dec-2016 hikes. 'Each quarter's decision' counts only quarters with "
        "a move.",
    )
    return label == FED_DEFINITIONS["cycle"]


def regime_controls(prefix: str, subject: str = "Dollar", fed_only: bool = False) -> RegimeSettings:
    """Render the regime controls with widget keys under prefix.

    fed_only shows just the Fed definition, response timing and pause length.
    """
    if fed_only:
        c_fed, c_resp = st.columns(2)
        basis_label = "Momentum"
    else:
        c_basis, c_fed, c_resp = st.columns(3)
        basis_label = c_basis.radio(
            "CPI & GDP regime basis",
            ["Momentum (YoY rate rising / falling)", "Direction (level up / down)"],
            horizontal=True,
            key=f"{prefix}_basis",
            help="The CPI index and GDP level rise in almost every quarter, so 'Direction' "
            "leaves very few Down quarters. 'Momentum' asks whether inflation / growth is "
            "accelerating.",
        )
    fed_cycle = _fed_controls(prefix, c_fed)
    response = c_resp.radio(
        f"{subject} response measured",
        ["Same quarter", "Next quarter"],
        horizontal=True,
        key=f"{prefix}_response",
        help="'Next quarter' compares a regime with the move in the following quarter, "
        "to check whether regimes lead FX.",
    )

    others = [k for k in DRIVERS if k != "FEDFUNDS"]
    deadband = {"FEDFUNDS": EFFECTIVE_FF_DEADBAND}
    max_pause = 4
    with st.expander("Regime thresholds"):
        cols = st.columns(1 if fed_only else len(others) + 1)
        max_pause = cols[0].number_input(
            "Longest pause inside a Fed cycle (quarters)",
            0,
            8,
            4,
            1,
            key=f"{prefix}_max_pause",
            disabled=not fed_cycle,
            help="Quarters with no move between two hikes (or two cuts) that still count as "
            "part of the hiking (cutting) cycle.",
        )
        if not fed_only:
            st.caption("Other variables: changes no bigger than the threshold are labelled Flat.")
            for col, k in zip(cols[1:], others, strict=True):
                var, key = MACRO_VARS[k], f"{prefix}_db_{k}"
                if var.kind == "rate":
                    deadband[k] = (
                        col.number_input(f"{var.short} (bps)", 0, 100, 0, 1, key=key) / 100
                    )
                else:
                    deadband[k] = col.number_input(
                        f"{var.short} (pp)", 0.0, 2.0, 0.0, 0.05, key=key
                    )
    for k in others:
        deadband.setdefault(k, 0.0)
    basis = "momentum" if basis_label.startswith("Momentum") else "direction"
    return RegimeSettings(basis, response, deadband, fed_cycle, int(max_pause))


def build_fed_regime(
    macro: dict[str, pd.Series], fed_target: pd.Series | None, settings: RegimeSettings
) -> pd.Series:
    """Hiking / On hold / Cutting per quarter (coded Up / Flat / Down)."""
    if fed_target is not None:
        return fed_regime(fed_target, settings.fed_cycle, settings.max_pause)
    # Fallback: effective rate with a band, since its monthly average drifts between moves.
    signal = regime_signals(macro, settings.basis)["FEDFUNDS"]
    moves = classify(signal, settings.deadband["FEDFUNDS"])
    return policy_cycle(moves, settings.max_pause) if settings.fed_cycle else moves


def build_regimes(
    macro: dict[str, pd.Series],
    settings: RegimeSettings,
    start,
    fed_target: pd.Series | None = None,
) -> pd.DataFrame:
    signals = regime_signals(macro, settings.basis)
    regimes = pd.DataFrame({k: classify(signals[k], settings.deadband[k]) for k in DRIVERS})
    regimes["FEDFUNDS"] = build_fed_regime(macro, fed_target, settings)
    return regimes[regimes.index >= pd.Timestamp(start)]


def response_returns(px: pd.Series, settings: RegimeSettings, start) -> pd.Series:
    """Quarterly % change, shifted back a quarter when measuring the next-quarter response."""
    ret = quarterly_returns(px)
    if settings.response == "Next quarter":
        ret = ret.shift(-1)
    return ret[ret.index >= pd.Timestamp(start)]
