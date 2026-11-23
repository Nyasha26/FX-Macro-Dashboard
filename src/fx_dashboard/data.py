"""Data loading from FRED (public CSV, no key) and Yahoo Finance."""

import io
import json
import os
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fx_dashboard.countries import Country, Source
from fx_dashboard.series import DOLLAR_INDEX, FED_TARGET_IDS, FX_PAIRS, MACRO_VARS, FxPair

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
# The official FRED API. The CSV download above needs no key but can hang when called from
# cloud servers (e.g. Streamlit Community Cloud), so the API is used whenever a key is set.
FRED_API_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
    "?series_id={series_id}&api_key={key}&file_type=json"
)


def _fetch(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fx-dashboard/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _load_fred_api(series_id: str, key: str) -> pd.Series:
    raw = _fetch(FRED_API_URL.format(series_id=series_id, key=key))
    obs = json.loads(raw)["observations"]
    s = pd.Series(
        pd.to_numeric([o["value"] for o in obs], errors="coerce"),  # "." marks a missing value
        index=pd.DatetimeIndex(pd.to_datetime([o["date"] for o in obs]), name="date"),
        name=series_id,
    )
    return s.dropna()


def load_fred_series(series_id: str) -> pd.Series:
    """Download one FRED series as a float Series indexed by date (missing values dropped).

    Uses the FRED API when FRED_API_KEY is set, falling back to the keyless CSV download.
    """
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        try:
            return _load_fred_api(series_id, key)
        except Exception:  # noqa: BLE001 - fall back to the CSV download
            pass
    df = pd.read_csv(io.BytesIO(_fetch(FRED_CSV_URL.format(series_id=series_id))))
    s = pd.Series(
        pd.to_numeric(df[series_id], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(df[df.columns[0]]), name="date"),
        name=series_id,
    )
    return s.dropna()


def load_yahoo_series(ticker: str) -> pd.Series:
    """Daily close from Yahoo Finance."""
    import yfinance as yf

    df = yf.download(ticker, start="1990-01-01", progress=False, auto_adjust=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.dropna().astype(float)
    if len(s) < 2:
        raise ValueError(f"Yahoo Finance returned no history for {ticker}")
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is not None else idx).rename("date")
    return s.rename(ticker)


def load_pair(pair: FxPair) -> pd.Series:
    """Load an FX pair quoted as foreign units per USD."""
    s = load_fred_series(pair.ticker) if pair.source == "fred" else load_yahoo_series(pair.ticker)
    if pair.invert:
        s = 1 / s
    return s.rename(pair.name)


BIS_URL = "https://stats.bis.org/api/v1/data/{id}/all?format=csv"
OECD_URL = "https://sdmx.oecd.org/public/rest/data/{id}?format=csvfile"
# A source whose latest observation is older than this is treated as discontinued.
MAX_AGE = {"M": pd.Timedelta(days=200), "Q": pd.Timedelta(days=330)}


def _sdmx_series(raw: bytes, name: str) -> pd.Series:
    """Parse an SDMX CSV (BIS or OECD) holding a single series."""
    df = pd.read_csv(io.BytesIO(raw), usecols=["TIME_PERIOD", "OBS_VALUE"])
    periods = df["TIME_PERIOD"].astype(str)
    if periods.str.contains("-Q").any():
        idx = pd.PeriodIndex(periods.str.replace("-Q", "Q"), freq="Q").to_timestamp()
    else:
        idx = pd.to_datetime(periods)
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").to_numpy(), index=idx)
    if s.index.has_duplicates:
        raise ValueError(f"{name}: query matched more than one series")
    return s.dropna().sort_index().rename_axis("date").rename(name)


BOE_CURVE_URL = (
    "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
    "glcnominalmonthedata.zip"
)
_boe_lock = threading.Lock()
_boe_cache: dict[str, pd.DataFrame] = {}


def parse_boe_curve(raw_zip: bytes) -> pd.DataFrame:
    """Month-end UK nominal spot curve (rows = dates, columns = maturity in years).

    The zip holds one workbook per period; each has a '4. spot curve' sheet whose
    'years:' row lists maturities above the data.
    """
    frames = []
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".xlsx")):
            df = pd.read_excel(io.BytesIO(z.read(name)), sheet_name="4. spot curve", header=None)
            hdr = df.index[df.iloc[:, 0].astype(str).str.strip().eq("years:")][0]
            maturities = pd.to_numeric(df.iloc[hdr, 1:], errors="coerce")
            body = df.iloc[hdr + 1 :]
            dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
            body = body[dates.notna()]
            frame = pd.DataFrame(
                body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(),
                index=pd.DatetimeIndex(dates[dates.notna()], name="date"),
                columns=maturities.to_numpy(),
            )
            frames.append(frame.loc[:, frame.columns.notna()])
    curve = pd.concat(frames).sort_index()
    return curve[~curve.index.duplicated(keep="last")]


def _boe_curve() -> pd.DataFrame:
    """Download once per day per process; the 2Y and the spread share one download."""
    day = pd.Timestamp.today().strftime("%Y-%m-%d")
    with _boe_lock:
        if day not in _boe_cache:
            _boe_cache.clear()
            _boe_cache[day] = parse_boe_curve(_fetch(BOE_CURVE_URL, timeout=120))
        return _boe_cache[day]


def load_boe_series(spec: str) -> pd.Series:
    """'spot:2' = 2Y spot yield; 'spread:10-2' = 10Y minus 2Y spot yield."""
    kind, tenor = spec.split(":")
    curve = _boe_curve()
    if kind == "spot":
        s = curve[float(tenor)]
    else:
        long_t, short_t = (float(t) for t in tenor.split("-"))
        s = curve[long_t] - curve[short_t]
    return s.dropna().rename(spec)


BOE_IADB_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
    "&Datefrom=01/Jan/1975&Dateto=now&SeriesCodes={code}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)


def load_boe_iadb_series(code: str) -> pd.Series:
    """One series from the Bank of England's database (e.g. IUDBEDR = Bank Rate, daily)."""
    raw = _fetch(BOE_IADB_URL.format(code=code), timeout=60)
    if raw.lstrip().startswith(b"<"):
        raise ValueError(f"Bank of England database returned an error page for {code}")
    df = pd.read_csv(io.BytesIO(raw))
    s = pd.Series(
        pd.to_numeric(df[code], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(df["DATE"], format="%d %b %Y"), name="date"),
    )
    return s.dropna().sort_index().rename(code)


def load_source(src: Source) -> pd.Series:
    if src.provider == "fred":
        return load_fred_series(src.id)
    if src.provider == "boe":
        return load_boe_series(src.id)
    if src.provider == "boe_iadb":
        return load_boe_iadb_series(src.id)
    if src.provider == "fred_spread":
        long_id, short_id = src.id.split("-")
        spread = (load_fred_series(long_id) - load_fred_series(short_id)).dropna()
        return spread.rename(src.id)
    url = (BIS_URL if src.provider == "bis" else OECD_URL).format(id=src.id)
    return _sdmx_series(_fetch(url, timeout=60), src.id)


def load_first_current(sources: tuple[Source, ...], freq: str) -> tuple[pd.Series, Source, str]:
    """Try sources in order and return the first that is still updating.

    If none is current, fall back to whichever has the latest data. Returns the series, the
    source used and a note explaining any fallback ("" when the first choice was used).
    """
    cutoff = pd.Timestamp.today().normalize() - MAX_AGE[freq]
    tried: list[tuple[pd.Series, Source]] = []
    notes = []
    for src in sources:
        try:
            s = load_source(src)
        except Exception as exc:  # noqa: BLE001 - try the next source
            notes.append(f"{src.describe()} failed ({type(exc).__name__})")
            continue
        if s.index.max() >= cutoff:
            return s, src, "; ".join(notes)
        notes.append(f"{src.describe()} stops at {s.index.max():%b %Y}")
        tried.append((s, src))
    if not tried:
        raise RuntimeError("; ".join(notes) or "no sources")
    s, src = max(tried, key=lambda t: t[0].index.max())
    return s, src, "; ".join(notes) + " (no current source; showing the latest available)"


def load_country(country: Country) -> dict:
    """Load every variable for one economy in parallel. Failures are collected, not raised."""
    out: dict = {"macro": {}, "sources": {}, "notes": {}, "errors": {}}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            key: pool.submit(load_first_current, srcs, country.vars[key].freq)
            for key, srcs in country.sources.items()
        }
        for key, fut in futures.items():
            try:
                s, src, note = fut.result()
            except Exception as exc:  # noqa: BLE001 - surface any loader failure in the UI
                out["errors"][key] = f"{type(exc).__name__}: {exc}"
                continue
            out["macro"][key] = s
            out["sources"][key] = src.describe()
            if not src.provider.startswith("fred"):
                if key == "FEDFUNDS":
                    note = (
                        "FRED has no official policy rate for this central bank, only a "
                        "monthly-average interbank rate that lags rate changes"
                        + (f"; {note}" if note else "")
                    )
                elif not note:
                    note = "Not available on FRED"
            if note:
                out["notes"][key] = note
    return out


def load_fed_target() -> pd.Series:
    """Daily Fed policy target: the single target until Dec 2008, then the range's upper bound."""
    old, upper = (load_fred_series(i) for i in FED_TARGET_IDS)
    s = pd.concat([old[old.index < upper.index.min()], upper]).sort_index()
    return s.rename("FEDTARGET")


def load_all() -> dict:
    """Fetch every series in parallel. Failures are collected rather than raised."""
    jobs = {("macro", k): (load_fred_series, v.fred_id) for k, v in MACRO_VARS.items()}
    jobs |= {("pairs", k): (load_pair, p) for k, p in FX_PAIRS.items()}
    jobs[("dollar", DOLLAR_INDEX)] = (load_fred_series, DOLLAR_INDEX)
    jobs[("fed_target", "Fed target rate")] = (lambda _: load_fed_target(), None)

    out: dict = {"macro": {}, "pairs": {}, "dollar": None, "fed_target": None, "errors": {}}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {key: pool.submit(fn, arg) for key, (fn, arg) in jobs.items()}
        for (group, name), fut in futures.items():
            try:
                s = fut.result()
            except Exception as exc:  # noqa: BLE001 - surface any loader failure in the UI
                out["errors"][name] = f"{type(exc).__name__}: {exc}"
                continue
            if group in ("dollar", "fed_target"):
                out[group] = s
            else:
                out[group][name] = s
    return out
