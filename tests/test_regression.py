import numpy as np
import pandas as pd
import pytest

from fx_dashboard.analysis import equal_and_opposite_pvalue, ols, rolling_ols


def _data(n=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1975-03-31", periods=n, freq="QE")
    X = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=idx, columns=["a", "b", "c"])
    y = 0.2 + 1.5 * X["a"] - 1.5 * X["b"] + rng.normal(0, 0.5, n)
    return y, X


def test_ols_recovers_coefficients_and_contributions_add_up():
    y, X = _data()
    reg = ols(y, X)
    assert reg.coef.loc["a", "coef"] == pytest.approx(1.5, abs=0.1)
    assert reg.coef.loc["b", "coef"] == pytest.approx(-1.5, abs=0.1)
    assert abs(reg.coef.loc["c", "t"]) < 3 and reg.coef.loc["a", "t"] > 10
    assert reg.coef.loc["a", "impact"] == pytest.approx(reg.coef.loc["a", "coef"] * X["a"].std())
    assert reg.r2 > 0.8 and reg.f_pvalue < 1e-6 and reg.nobs == len(y)
    total = reg.contributions().sum(axis=1) + reg.const + reg.resid
    assert np.allclose(total, reg.y)


def test_ols_drops_constant_columns_and_needs_enough_rows():
    y, X = _data(30)
    X["flat"] = 1.0
    reg = ols(y, X)
    assert "flat" not in reg.coef.index
    assert ols(y.iloc[:6], X.iloc[:6]) is None


def test_equal_and_opposite_test():
    y, X = _data()
    reg = ols(y, X)
    assert equal_and_opposite_pvalue(reg, [("a", "b")]) > 0.01  # true: a = -b
    assert equal_and_opposite_pvalue(reg, [("a", "c")]) < 0.001  # false
    assert np.isnan(equal_and_opposite_pvalue(reg, [("x", "y")]))


def test_rolling_ols_windows():
    y, X = _data(60)
    roll = rolling_ols(y, X, 20)
    assert len(roll["coef"]) == 41 and list(roll["coef"].columns) == ["a", "b", "c"]
    assert roll["coef"]["a"].between(0.8, 2.2).all()
