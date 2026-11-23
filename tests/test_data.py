import io
import zipfile

import pandas as pd
import pytest

from fx_dashboard import data
from fx_dashboard.countries import COUNTRIES, Source
from fx_dashboard.series import MACRO_VARS


def test_sdmx_parses_monthly_and_quarterly_periods():
    monthly = b"FREQ,TIME_PERIOD,OBS_VALUE\nM,2026-08,1.5\nM,2026-09,1.75\nM,2026-10,\n"
    s = data._sdmx_series(monthly, "x")
    assert list(s.index) == [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")]
    quarterly = b"TIME_PERIOD,OBS_VALUE,REF_AREA\n2026-Q1,100,NZL\n2026-Q2,101,NZL\n"
    q = data._sdmx_series(quarterly, "y")
    assert q.index[-1] == pd.Timestamp("2026-04-01") and q.iloc[-1] == 101


def test_sdmx_rejects_several_series():
    raw = b"TIME_PERIOD,OBS_VALUE\n2026-01,1\n2026-01,2\n"
    with pytest.raises(ValueError):
        data._sdmx_series(raw, "z")


def test_fred_uses_api_when_key_set(monkeypatch):
    raw = (
        b'{"observations": [{"date": "2026-10-01", "value": "3.5"},'
        b' {"date": "2026-11-01", "value": "."}]}'
    )
    urls = []

    def fake_fetch(url, timeout=30, retries=3):
        urls.append(url)
        return raw

    monkeypatch.setattr(data, "_fetch", fake_fetch)
    monkeypatch.setenv("FRED_API_KEY", "abc")
    s = data.load_fred_series("DGS2")
    assert "api.stlouisfed.org" in urls[0] and len(s) == 1 and s.iloc[-1] == 3.5
    # without a key the keyless CSV download is used
    monkeypatch.delenv("FRED_API_KEY")
    monkeypatch.setattr(
        data, "_fetch", lambda url, **kw: b"observation_date,DGS2\n2026-10-01,3.5\n"
    )
    assert data.load_fred_series("DGS2").iloc[-1] == 3.5


def test_fred_reports_both_errors_when_api_and_csv_fail(monkeypatch):
    def boom(url, **kw):
        raise TimeoutError("timed out")

    monkeypatch.setattr(data, "_fetch", boom)
    monkeypatch.setenv("FRED_API_KEY", "abc")
    with pytest.raises(RuntimeError, match="FRED API: TimeoutError.*FRED CSV: TimeoutError"):
        data.load_fred_series("DGS2")


def _series(last: str) -> pd.Series:
    return pd.Series(1.0, index=pd.date_range(end=last, periods=24, freq="MS"))


def test_load_first_current_skips_stale_fred(monkeypatch):
    today = pd.Timestamp.today().normalize()
    fresh = (today - pd.DateOffset(months=1)).strftime("%Y-%m-01")
    stale = (today - pd.DateOffset(years=2)).strftime("%Y-%m-01")
    fake = {"OLD": _series(stale), "NEW": _series(fresh)}
    monkeypatch.setattr(data, "load_source", lambda src: fake[src.id])
    s, src, note = data.load_first_current((Source("fred", "OLD"), Source("bis", "NEW")), "M")
    assert src.id == "NEW" and "stops at" in note
    # the first source wins whenever it is current
    s, src, note = data.load_first_current((Source("fred", "NEW"), Source("bis", "OLD")), "M")
    assert src.id == "NEW" and note == ""


def test_load_first_current_keeps_latest_when_all_stale(monkeypatch):
    fake = {"A": _series("2020-01-01"), "B": _series("2022-01-01")}
    monkeypatch.setattr(data, "load_source", lambda src: fake[src.id])
    s, src, note = data.load_first_current((Source("fred", "A"), Source("bis", "B")), "M")
    assert src.id == "B" and "no current source" in note


def test_every_country_covers_the_us_variables_with_fred_first():
    for c in COUNTRIES.values():
        assert list(c.vars) == list(MACRO_VARS)
        assert set(c.sources) == set(MACRO_VARS)
        for key, srcs in c.sources.items():
            first = srcs[0].provider
            if key == "FEDFUNDS":
                # official policy rates: FRED's ECB rate, otherwise the BoE / BIS
                assert first in ("fred", "bis", "boe_iadb"), c.code
                assert not any(s.id.startswith("IRSTCI01") for s in srcs), c.code
            else:
                # FRED first everywhere else, except the UK gilt curve, which FRED lacks
                assert first.startswith("fred") or (c.code == "GB" and first == "boe"), (
                    c.code,
                    key,
                )
    assert COUNTRIES["GB"].sources["FEDFUNDS"][0] == Source("boe_iadb", "IUDBEDR")


def test_load_boe_iadb_series(monkeypatch):
    raw = b"DATE,IUDBEDR\n18 Dec 2025,3.75\n05 Nov 2026,4\n19 Nov 2026,4\n"
    monkeypatch.setattr(data, "_fetch", lambda url, timeout=30: raw)
    s = data.load_boe_iadb_series("IUDBEDR")
    assert s.index[-1] == pd.Timestamp("2026-11-19") and s.iloc[-1] == 4.0
    monkeypatch.setattr(data, "_fetch", lambda url, timeout=30: b"<!DOCTYPE html>")
    with pytest.raises(ValueError):
        data.load_boe_iadb_series("BAD")


def _boe_zip() -> bytes:
    rows = [
        [None, "UK nominal spot curve", None, None],
        [None, None, None, None],
        ["Maturity", None, None, None],
        ["years:", 2.0, 10.0, None],
        [pd.Timestamp("2026-09-30"), 4.66, 5.20, None],
        [pd.Timestamp("2026-10-30"), 4.68, 5.25, None],
    ]
    book = io.BytesIO()
    pd.DataFrame(rows).to_excel(book, sheet_name="4. spot curve", header=False, index=False)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("GLC Nominal month end data_2025 to present.xlsx", book.getvalue())
    return out.getvalue()


def test_parse_boe_curve_and_spread(monkeypatch):
    curve = data.parse_boe_curve(_boe_zip())
    assert list(curve.columns) == [2.0, 10.0]
    assert curve.loc["2026-10-30", 2.0] == pytest.approx(4.68)
    monkeypatch.setattr(data, "_boe_curve", lambda: curve)
    assert data.load_boe_series("spot:2").iloc[-1] == pytest.approx(4.68)
    assert data.load_boe_series("spread:10-2").iloc[-1] == pytest.approx(0.57)
