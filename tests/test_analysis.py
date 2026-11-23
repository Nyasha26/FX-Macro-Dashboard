import numpy as np
import pandas as pd
import pytest

from fx_dashboard.analysis import (
    DRIVERS,
    classify,
    episode_returns,
    fed_regime,
    fit_sensitivity,
    macro_drivers,
    pair_regime_stats,
    policy_cycle,
    regime_blocks,
    regime_matrix,
    regime_signals,
    regime_stats,
    scenario_impact,
    to_quarterly,
    trailing_returns,
)
from fx_dashboard.series import MACRO_VARS


def test_to_quarterly_drops_unfinished_quarter_for_daily_data():
    s = pd.Series(1.0, index=pd.bdate_range("2024-01-01", "2024-11-15"))
    q = to_quarterly(s)
    assert q.index[-1] == pd.Timestamp("2024-09-30")


def test_to_quarterly_keeps_quarterly_data_dated_at_quarter_start():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01"]))
    q = to_quarterly(s)
    assert list(q) == [1.0, 2.0, 3.0]
    assert q.index[-1] == pd.Timestamp("2024-09-30")


def test_to_quarterly_monthly_needs_last_month_of_quarter():
    s = pd.Series(1.0, index=pd.date_range("2024-01-01", "2024-10-01", freq="MS"))
    assert to_quarterly(s).index[-1] == pd.Timestamp("2024-09-30")


def test_classify_with_deadband():
    sig = pd.Series([0.5, 0.05, -0.05, -0.5, np.nan])
    out = classify(sig, deadband=0.1)
    assert list(out[:4]) == ["Up", "Flat", "Flat", "Down"]
    assert pd.isna(out.iloc[4])


def test_regime_stats_and_matrix():
    idx = pd.period_range("2020Q1", periods=4, freq="Q").to_timestamp(how="end").normalize()
    ret = pd.Series([1.0, 3.0, -2.0, -4.0], index=idx)
    regimes = pd.DataFrame(
        {"A": ["Up", "Up", "Down", "Down"], "B": ["Up", "Down", "Up", "Down"]}, index=idx
    )
    stats = regime_stats(ret, regimes).set_index(["variable", "regime"])
    assert stats.loc[("A", "Up"), "mean"] == pytest.approx(2.0)
    assert stats.loc[("A", "Down"), "hit_rate"] == 0
    assert stats.loc[("A", "Up"), "n"] == 2
    mean, count = regime_matrix(ret, regimes, "A", "B")
    assert mean.loc["Up", "Down"] == pytest.approx(3.0)
    assert count.to_numpy().sum() == 4


def test_pair_regime_stats_stacks_pairs():
    idx = pd.period_range("2020Q1", periods=4, freq="Q").to_timestamp(how="end").normalize()
    regimes = pd.DataFrame({"A": ["Up", "Up", "Down", "Down"]}, index=idx)
    rets = {
        "X": pd.Series([1.0, 3.0, -2.0, -4.0], index=idx),
        "Y": pd.Series([-1.0, -1.0, 5.0, 5.0], index=idx),
    }
    out = pair_regime_stats(rets, regimes).set_index(["pair", "regime"])
    assert out.loc[("X", "Up"), "mean"] == pytest.approx(2.0)
    assert out.loc[("Y", "Down"), "mean"] == pytest.approx(5.0)
    assert len(out) == 4


def test_regime_blocks_merge_runs():
    idx = pd.period_range("2020Q1", periods=4, freq="Q").to_timestamp(how="end").normalize()
    blocks = regime_blocks(pd.Series(["Up", "Up", "Down", "Up"], index=idx))
    assert [b[2] for b in blocks] == ["Up", "Down", "Up"]
    assert blocks[0][0] == pd.Timestamp("2020-01-01")


def test_fit_sensitivity_recovers_betas_and_scenario_is_linear():
    rng = np.random.default_rng(1)
    n = 200
    x = pd.DataFrame(rng.normal(size=(n, 2)), columns=["FEDFUNDS", "GDP"])
    y = 0.5 + 2.0 * x["FEDFUNDS"] - 1.0 * x["GDP"] + rng.normal(scale=0.01, size=n)
    sens = fit_sensitivity(y, x)
    assert sens is not None
    assert sens.beta["FEDFUNDS"] == pytest.approx(2.0, abs=0.01)
    assert sens.beta["GDP"] == pytest.approx(-1.0, abs=0.01)
    contrib, total, se = scenario_impact(sens, {"FEDFUNDS": 0.02, "GDP": 1.0})
    assert total == pytest.approx(2.0 * 0.02 - 1.0, abs=0.01)
    assert contrib.sum() == pytest.approx(total)
    assert se > 0


def _quarterly_macro():
    """Synthetic quarterly macro data for every driver, dated like FRED (quarter start)."""
    idx = pd.date_range("2015-01-01", periods=24, freq="QS")
    rng = np.random.default_rng(2)
    rate = pd.Series(np.cumsum(rng.normal(size=24)), index=idx)
    index = pd.Series(100 * np.cumprod(1 + rng.normal(0.01, 0.01, size=24)), index=idx)
    return {k: (rate if MACRO_VARS[k].kind == "rate" else index) for k in DRIVERS}


def test_macro_drivers_and_signals_cover_every_driver():
    macro = _quarterly_macro()
    drivers = macro_drivers(macro)
    assert list(drivers.columns) == DRIVERS
    # rates and spreads are level changes; indices are % changes
    q_rate = macro["T10Y2Y"].resample("QE").last()
    assert drivers["T10Y2Y"].iloc[-1] == pytest.approx(q_rate.iloc[-1] - q_rate.iloc[-2])
    q_idx = macro["GDP"].resample("QE").last()
    assert drivers["GDP"].iloc[-1] == pytest.approx((q_idx.iloc[-1] / q_idx.iloc[-2] - 1) * 100)
    signals = regime_signals(macro, "momentum")
    assert list(signals.columns) == DRIVERS
    assert signals["T10Y2Y"].equals(drivers["T10Y2Y"])


def test_fit_sensitivity_requires_min_obs():
    x = pd.DataFrame({"FEDFUNDS": [1.0, 2.0, 3.0]})
    assert fit_sensitivity(pd.Series([1.0, 2.0, 3.0]), x) is None


def test_trailing_returns():
    s = pd.Series(np.linspace(100, 110, 400), index=pd.bdate_range("2024-01-01", periods=400))
    tr = trailing_returns(s)
    assert tr["1M"] > 0 and tr["1Y"] > tr["3M"] > 0


def _q(n, start="2015Q1"):
    return pd.period_range(start, periods=n, freq="Q").to_timestamp(how="end").normalize()


def test_policy_cycle_fills_pauses_between_same_direction_moves():
    moves = pd.Series(["Up", "Flat", "Flat", "Up", "Flat", "Down", "Flat", "Flat"], index=_q(8))
    out = policy_cycle(moves, max_pause=4)
    # pause between two hikes joins the cycle; hold between a hike and a cut, and after
    # the last move, stays on hold
    assert list(out) == ["Up", "Up", "Up", "Up", "Flat", "Down", "Flat", "Flat"]
    assert list(policy_cycle(moves, max_pause=1))[:4] == ["Up", "Flat", "Flat", "Up"]


def test_fed_regime_from_daily_target():
    days = pd.date_range("2015-01-01", "2016-12-31", freq="D")
    target = pd.Series(0.25, index=days)
    target[target.index >= "2015-12-16"] = 0.50  # Q4 2015 hike
    target[target.index >= "2016-12-14"] = 0.75  # Q4 2016 hike
    out = fed_regime(target, cycle=False)
    assert out[pd.Timestamp("2015-12-31")] == "Up"
    assert out[pd.Timestamp("2016-06-30")] == "Flat"
    cyc = fed_regime(target, cycle=True, max_pause=4)
    assert (cyc["2016-03-31":"2016-12-31"] == "Up").all()


def test_regime_stats_annualised_and_episodes():
    idx = _q(4)
    ret = pd.Series([10.0, 10.0, -5.0, 2.0], index=idx)
    regime = pd.Series(["Up", "Up", "Down", "Down"], index=idx)
    stats = regime_stats(ret, regime.to_frame("F")).set_index("regime")
    assert stats.loc["Up", "ann"] == pytest.approx((1.1**4 - 1) * 100)
    ep = episode_returns({"X": ret}, regime)
    assert list(ep["regime"]) == ["Up", "Down"]
    assert ep.loc[0, "X"] == pytest.approx(21.0)
    assert ep.loc[1, "quarters"] == 2
