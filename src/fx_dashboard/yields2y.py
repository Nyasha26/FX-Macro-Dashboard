"""2-year government bond yields for Germany, Japan and the UK, from downloaded files.

FRED has no 2Y government yield outside the US, so these come from each issuer's own
download, which the dashboard ships as one small bundled CSV (resources/yields_2y.csv):

  DE  Bundesbank: daily yield of the current 2Y Federal Treasury note (BBSSY...WT0202.A)
  JP  Ministry of Finance Japan: JGB benchmark yields, 2Y column
  GB  Bank of England: daily nominal gilt spot curve, 2.0-year point

To refresh, download new copies of the three files into one folder and run
    python -m fx_dashboard.yields2y "<folder>"
which rewrites the bundled CSV (commit and push it to update the online app).
"""

import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

BUNDLED = Path(__file__).parent / "resources" / "yields_2y.csv"
LABELS = {"DE": "2Y Bund yield", "JP": "2Y JGB yield", "GB": "2Y gilt yield (spot curve)"}
SOURCES = {
    "DE": "Bundesbank, current 2Y Federal Treasury note (daily)",
    "JP": "Ministry of Finance Japan, JGB 2Y (daily)",
    "GB": "Bank of England, nominal spot curve at 2 years (daily)",
}


def _clean(dates: pd.Series, values: pd.Series, name: str) -> pd.Series:
    s = pd.Series(
        pd.to_numeric(values, errors="coerce").to_numpy(),  # "." / "-" mark missing days
        index=pd.DatetimeIndex(dates, name="date"),
        name=name,
    )
    s = s[s.index.notna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def parse_bundesbank(path: Path) -> pd.Series:
    """Bundesbank time-series CSV: metadata rows, then date, value, flag."""
    df = pd.read_csv(path, header=None, dtype=str, usecols=[0, 1])
    dates = pd.to_datetime(df[0], format="%m/%d/%Y", errors="coerce")
    if dates.notna().sum() == 0:  # the Bundesbank's own export uses ISO dates
        dates = pd.to_datetime(df[0], format="%Y-%m-%d", errors="coerce")
    keep = dates.notna()
    return _clean(dates[keep], df.loc[keep, 1], "DE")


def parse_mof_jgb(path: Path, column: str = "2Y") -> pd.Series:
    """Ministry of Finance JGB yields: a Date column and one column per tenor."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    dates = pd.to_datetime(df[df.columns[0]], format="mixed", errors="coerce")
    return _clean(dates, df[column], "JP")


def parse_boe_spot_curve(path: Path, tenor: float = 2.0) -> pd.Series:
    """Bank of England daily spot curve: a 'years:' row of maturities above dated rows."""
    df = pd.read_csv(path, header=None, dtype=str)
    hdr = df.index[df[0].astype(str).str.strip().eq("years:")][0]
    maturities = pd.to_numeric(df.iloc[hdr], errors="coerce")
    col = maturities.index[maturities == tenor][0]
    body = df.iloc[hdr + 1 :]
    dates = pd.to_datetime(body[0], format="%d %b %y", errors="coerce")
    return _clean(dates, body[col], "GB")


RAW_FILES: dict[str, tuple[str, Callable[[Path], pd.Series]]] = {
    "DE": ("2yr Bund Yields.csv", parse_bundesbank),
    "JP": ("JGB_YIELDS .csv", parse_mof_jgb),
    "GB": ("Gilt spot curve.csv", parse_boe_spot_curve),
}


def build(folder: Path, out: Path = BUNDLED) -> pd.DataFrame:
    """Parse the downloaded files in folder and write the bundled CSV."""
    cols = {code: parse(Path(folder) / name) for code, (name, parse) in RAW_FILES.items()}
    df = pd.DataFrame(cols).sort_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, float_format="%.4g", date_format="%Y-%m-%d")
    return df


def load(path: Path = BUNDLED) -> dict[str, pd.Series]:
    """Bundled 2Y yields by ISO country code (empty if the file is missing)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return {c: df[c].dropna().rename(c) for c in df.columns}


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    built = build(folder)
    for code in built.columns:
        s = built[code].dropna()
        print(f"{code}: {s.index.min():%Y-%m-%d} to {s.index.max():%Y-%m-%d}, {len(s)} days")
    print(f"Wrote {BUNDLED}")
