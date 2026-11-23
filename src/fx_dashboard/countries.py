"""Macro series for the other G10 economies shown on the Macro Variables tab.

Each variable lists its sources in order of preference. FRED comes first wherever it carries
the series; the loader falls back to the BIS or OECD when the FRED series has stopped
updating (several OECD-derived FRED series were frozen in 2021-2025). Policy rates are the
exception: outside the euro area FRED only has monthly-average interbank rates, so official
central-bank rates come from the BIS (and the Bank of England for Bank Rate).

Variables use the same keys as the US catalogue (MACRO_VARS) so toggles and colours line up,
but two are proxies because FRED carries no 2Y government yield outside the US:
  DGS2   -> 3-month interbank rate
  T10Y2Y -> 10-year government yield minus the 3-month interbank rate
"""

from dataclasses import dataclass
from typing import Literal

from fx_dashboard.series import MacroVar

Provider = Literal["fred", "fred_spread", "bis", "oecd", "boe", "boe_iadb"]


@dataclass(frozen=True)
class Source:
    provider: Provider
    id: str  # FRED id; "LONG-SHORT" FRED ids for fred_spread; BIS / OECD SDMX "flow/key"

    def describe(self) -> str:
        if self.provider == "fred":
            return f"FRED {self.id}"
        if self.provider == "fred_spread":
            long_id, short_id = self.id.split("-")
            return f"FRED {long_id} − {short_id}"
        if self.provider == "bis":
            return f"BIS {self.id}"
        if self.provider == "boe_iadb":
            return f"Bank of England database {self.id}"
        if self.provider == "boe":
            kind, tenor = self.id.split(":")
            what = (
                f"{tenor}Y spot yield" if kind == "spot" else f"{tenor.replace('-', 'Y − ')}Y spot"
            )
            return f"Bank of England gilt curve, {what}"
        return f"OECD {self.id.split('/')[0].split(',')[1]}"


@dataclass(frozen=True)
class Country:
    code: str  # ISO 2-letter code, as FRED and the BIS use
    name: str
    currency: str
    policy_name: str
    vars: dict[str, MacroVar]
    sources: dict[str, tuple[Source, ...]]


_EURO = {"FR", "DE", "IT"}
# Australia and New Zealand publish CPI quarterly.
_QUARTERLY_CPI = {"AU", "NZ"}
_HICP = {"SE", "NO", "FR", "DE", "IT", "CH"}  # FRED carries Eurostat HICP for these
# FRED CPI series (OECD MEI) for the rest; frozen since 2021-25, so the BIS usually wins.
_FRED_CPI = {
    "AU": "AUSCPIALLQINMEI",
    "NZ": "NZLCPIALLQINMEI",
    "JP": "JPNCPIALLMINMEI",
    "CA": "CANCPIALLMINMEI",
    "GB": "GBRCPIALLMINMEI",
}
# Real GDP: IMF-sourced FRED series where they exist, Eurostat ones otherwise.
_FRED_GDP = {
    "AU": ("NGDPRSAXDCAUQ",),
    "SE": ("CLVMNACSCAB1GQSE",),
    "NO": ("CLVMNACSCAB1GQNO",),
    "FR": ("NGDPRSAXDCFRQ", "CLVMNACSCAB1GQFR"),
    "DE": ("NGDPRSAXDCDEQ", "CLVMNACSCAB1GQDE"),
    "IT": ("NGDPRSAXDCITQ", "CLVMNACSCAB1GQIT"),
    "NZ": ("NAEXKP01NZQ189S",),
    "CH": ("CLVMNACSCAB1GQCH",),
    "JP": ("NGDPRSAXDCJPQ",),
    "CA": ("NGDPRSAXDCCAQ",),
    "GB": ("NGDPRSAXDCGBQ",),
}
_OECD_GDP = {"NZ": "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.1/Q.Y.NZL.S1..B1GQ._Z...XDC.LR.."}
# The UK has no current 3M rate on FRED (it stops in Jan 2026), but the Bank of England
# publishes the whole gilt curve, so the UK gets a true 2Y yield and 10Y-2Y spread like the US.
_BOE_CURVE = {"GB"}

_ECONOMIES = [
    ("AU", "Australia", "AUD", "RBA cash rate"),
    ("CA", "Canada", "CAD", "BoC policy rate"),
    ("FR", "France", "EUR", "ECB deposit rate"),
    ("DE", "Germany", "EUR", "ECB deposit rate"),
    ("IT", "Italy", "EUR", "ECB deposit rate"),
    ("JP", "Japan", "JPY", "BoJ policy rate"),
    ("NZ", "New Zealand", "NZD", "RBNZ official cash rate"),
    ("NO", "Norway", "NOK", "Norges Bank policy rate"),
    ("SE", "Sweden", "SEK", "Riksbank policy rate"),
    ("CH", "Switzerland", "CHF", "SNB policy rate"),
    ("GB", "United Kingdom", "GBP", "BoE Bank Rate"),
]


def _country(code: str, name: str, currency: str, policy_name: str) -> Country:
    cpi_freq = "Q" if code in _QUARTERLY_CPI else "M"
    cpi_label = "HICP (all items, NSA)" if code in _HICP else "CPI (all items, NSA)"
    vars_ = {
        v.key: v
        for v in [
            MacroVar("DGS2", "", "3M interbank rate", "3M rate", "%", "rate", "M"),
            MacroVar(
                "T10Y2Y", "", "10Y yield – 3M rate spread", "10Y–3M spread", "pp", "rate", "M"
            ),
            MacroVar("CPI", "", cpi_label, "CPI", "index", "index", cpi_freq),
            MacroVar("FEDFUNDS", "", policy_name, "Policy rate", "%", "rate", "M"),
            MacroVar("GDP", "", "Real GDP", "Real GDP", "real, national ccy, SA", "index", "Q"),
        ]
    }

    if code in _EURO:
        policy = (Source("fred", "ECBDFR"),)
    else:
        # FRED has no current policy rate for these central banks. Its overnight interbank
        # series (IRSTCI01) is a monthly average that misses recent moves (e.g. 3.73% when
        # Bank Rate was 4%), so use the official rate: daily from the BIS, month-end as backup.
        policy = (Source("bis", f"WS_CBPOL/D.{code}"), Source("bis", f"WS_CBPOL/M.{code}"))
        if code == "GB":
            policy = (Source("boe_iadb", "IUDBEDR"),) + policy
    if code in _HICP:
        cpi = (Source("fred", f"CP0000{code}M086NEST"),)
    else:
        cpi = (Source("fred", _FRED_CPI[code]),)
    cpi += (Source("bis", f"WS_LONG_CPI/M.{code}.628"),)
    gdp = tuple(Source("fred", i) for i in _FRED_GDP[code])
    if code in _OECD_GDP:
        gdp += (Source("oecd", _OECD_GDP[code]),)

    sources = {
        "DGS2": (Source("fred", f"IR3TIB01{code}M156N"),),
        "T10Y2Y": (Source("fred_spread", f"IRLTLT01{code}M156N-IR3TIB01{code}M156N"),),
        "CPI": cpi,
        "FEDFUNDS": policy,
        "GDP": gdp,
    }
    if code in _BOE_CURVE:
        vars_["DGS2"] = MacroVar("DGS2", "", "2Y gilt yield", "2Y yield", "%", "rate", "M")
        vars_["T10Y2Y"] = MacroVar(
            "T10Y2Y", "", "10Y–2Y gilt spread", "10Y–2Y spread", "pp", "rate", "M"
        )
        sources["DGS2"] = (Source("boe", "spot:2"),)
        sources["T10Y2Y"] = (Source("boe", "spread:10-2"),)
    return Country(code, name, currency, policy_name, vars_, sources)


COUNTRIES: dict[str, Country] = {c[0]: _country(*c) for c in _ECONOMIES}
