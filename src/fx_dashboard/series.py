"""Catalogue of the series the dashboard uses.

Every FX pair is quoted with USD as the base currency (foreign units per 1 USD),
so a rising pair always means a stronger dollar, matching the dollar index.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MacroVar:
    key: str
    fred_id: str
    label: str
    short: str
    unit: str
    kind: Literal["rate", "index"]  # rates change in pp; indices change in %
    freq: Literal["M", "Q"]  # display frequency


MACRO_VARS: dict[str, MacroVar] = {
    v.key: v
    for v in [
        MacroVar("DGS2", "DGS2", "2Y Treasury yield", "2Y yield", "%", "rate", "M"),
        MacroVar("T10Y2Y", "T10Y2Y", "10Y–2Y Treasury spread", "10Y–2Y spread", "pp", "rate", "M"),
        MacroVar("CPI", "CPIAUCSL", "CPI (all items, SA)", "CPI", "index", "index", "M"),
        MacroVar("FEDFUNDS", "FEDFUNDS", "Fed funds rate", "Fed funds", "%", "rate", "M"),
        MacroVar("GDP", "GDPC1", "Real GDP", "Real GDP", "bn chained $, SAAR", "index", "Q"),
    ]
}

# Nominal Broad U.S. Dollar Index (Fed, Jan 2006 = 100).
DOLLAR_INDEX = "DTWEXBGS"

# Fed policy target, used to date hikes and cuts: a single target rate until 15 Dec 2008,
# then the upper bound of the target range.
FED_TARGET_IDS = ("DFEDTAR", "DFEDTARU")


@dataclass(frozen=True)
class FxPair:
    name: str
    source: Literal["fred", "yahoo"]
    ticker: str
    invert: bool = False  # True when the source quotes USD per foreign unit
    note: str = ""


FX_PAIRS: dict[str, FxPair] = {
    p.name: p
    for p in [
        FxPair("USDEUR", "fred", "DEXUSEU", invert=True, note="Inverted from FRED EUR/USD"),
        FxPair("USDJPY", "fred", "DEXJPUS"),
        FxPair("USDCHF", "fred", "DEXSZUS"),
        FxPair("USDGBP", "fred", "DEXUSUK", invert=True, note="Inverted from FRED GBP/USD"),
        FxPair("USDZAR", "fred", "DEXSFUS"),
        FxPair("USDBRL", "fred", "DEXBZUS"),
        FxPair(
            "USDCNH",
            "fred",
            "DEXCHUS",
            note="Onshore USD/CNY used as a proxy: no free offshore CNH history is available",
        ),
        FxPair("USDMXN", "fred", "DEXMXUS"),
        FxPair("USDCLP", "yahoo", "CLP=X", note="Yahoo Finance daily close"),
    ]
}
