import numpy as np
import pandas as pd
import pytest

from fx_dashboard import yields2y
from fx_dashboard.analysis import gap_betas, gap_implied_spot, weekly_spot_and_gap


def test_parse_bundesbank(tmp_path):
    p = tmp_path / "bund.csv"
    p.write_text(
        ",BBSSY.D.REN.EUR.A610.000000WT0202.A,BBSSY.D.REN.EUR.A610.000000WT0202.A_FLAGS\n"
        ",Daily yield of the current (two-year) Federal Treasury notes,\n"
        "unit,PROZENT,\n"
        "last update,11/20/2026 12:29,\n"
        "1/2/2014,0.21,\n"
        "1/4/2014,.,No value available\n"
        "11/20/2026,3.08,\n"
    )
    s = yields2y.parse_bundesbank(p)
    assert list(s.index) == [pd.Timestamp("2014-01-02"), pd.Timestamp("2026-11-20")]
    assert s.iloc[-1] == pytest.approx(3.08)


def test_parse_mof_jgb(tmp_path):
    p = tmp_path / "jgb.csv"
    p.write_text("Date,2Y\n9/24/1974,9.362\n1/4/1993,-\n10/30/2026,1.875\n")
    s = yields2y.parse_mof_jgb(p)
    assert len(s) == 2 and s.loc["2026-10-30"] == pytest.approx(1.875)


def test_parse_boe_spot_curve(tmp_path):
    p = tmp_path / "gilt.csv"
    p.write_text(
        ",UK nominal spot curve,,,\n"
        ",,,,\n"
        "Maturity,,,,\n"
        "years:,0.5,1.0,1.5,2.0\n"
        ",,,,\n"
        "01 Jan 79,,,,\n"
        "02 Jan 79,,11.45,11.73,11.96\n"
        "30 Oct 26,4.22,4.43,4.58,4.68\n"
    )
    s = yields2y.parse_boe_spot_curve(p)
    assert s.index[0] == pd.Timestamp("1979-01-02") and s.iloc[0] == pytest.approx(11.96)
    assert s.index[-1] == pd.Timestamp("2026-10-30") and s.iloc[-1] == pytest.approx(4.68)


def test_bundled_file_covers_germany_japan_uk():
    y = yields2y.load()
    assert set(y) == {"DE", "JP", "GB"}
    for s in y.values():
        assert s.index.is_monotonic_increasing and s.between(-2, 20).all()


def test_gap_betas_recovers_slope():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=900)
    gap = pd.Series(np.cumsum(rng.normal(0, 0.03, len(idx))) + 2, index=idx)
    # spot moves 0.5% for every 10bp (0.1pp) move in the gap, plus noise
    log_spot = np.log(100) + 0.05 * (gap - gap.iloc[0]) + rng.normal(0, 0.0005, len(idx))
    spot = pd.Series(np.exp(log_spot), index=idx)
    weekly = weekly_spot_and_gap(spot, gap)
    b = gap_betas(weekly, {"all": None, "1y": 52}).set_index("window")
    assert b.loc["all", "beta_10bp"] == pytest.approx(0.5, abs=0.05)
    assert b.loc["1y", "weeks"] <= 52 and b.loc["all", "t"] > 10
    implied, slope, r2 = gap_implied_spot(weekly, 52)
    assert slope > 0 and r2 > 0.8 and implied == pytest.approx(weekly["spot"].iloc[-1], rel=0.02)
