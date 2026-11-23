"""Analysis: quarterly macro drivers, regime classification and FX sensitivity scenarios.

Everything runs at quarterly frequency because GDP is only published quarterly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fx_dashboard.series import MACRO_VARS

# Order sets chart stacking and colour slots; append new variables at the end.
DRIVERS = ["FEDFUNDS", "DGS2", "CPI", "GDP", "T10Y2Y"]
REGIME_ORDER = ["Up", "Flat", "Down"]


def pct(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods, fill_method=None) * 100


def to_quarterly(s: pd.Series) -> pd.Series:
    """End-of-quarter values, dropping a final quarter the series has not finished yet."""
    q = s.resample("QE").last()
    if len(q) < 2:
        return q
    spacing = s.index.to_series().diff().median()
    last_obs, q_end = s.index.max(), q.index[-1]
    if spacing >= pd.Timedelta(days=80):  # quarterly data: every observation is a full quarter
        finished = True
    elif spacing >= pd.Timedelta(days=20):  # monthly data: need the quarter's last month
        finished = last_obs.month == q_end.month
    else:  # daily data: need observations up to (nearly) quarter end
        finished = last_obs >= q_end - pd.Timedelta(days=5)
    return q if finished else q.iloc[:-1]


def quarterly_returns(px: pd.Series) -> pd.Series:
    """Quarter-on-quarter % change of a price series."""
    return pct(to_quarterly(px))


def macro_drivers(macro: dict[str, pd.Series]) -> pd.DataFrame:
    """Quarterly macro changes used as regressors, in the units the scenario shocks use.

    Rates and spreads (FEDFUNDS, DGS2, T10Y2Y): change in level (percentage points).
    Indices (CPI, GDP): quarter-on-quarter % change (not annualised).
    """
    q = {k: to_quarterly(macro[k]) for k in DRIVERS}
    return pd.DataFrame(
        {k: q[k].diff() if MACRO_VARS[k].kind == "rate" else pct(q[k]) for k in DRIVERS}
    )


def regime_signals(macro: dict[str, pd.Series], basis: str = "momentum") -> pd.DataFrame:
    """Signed quarterly signal whose sign defines each variable's Up/Down regime.

    Rates and spreads are always classified by the change in their level.
    For indices (CPI, GDP):
      basis="momentum": change in the year-on-year rate (inflation / growth accelerating?)
      basis="direction": quarter-on-quarter % change of the level (index rising?)
    """
    out = {}
    for k in DRIVERS:
        q = to_quarterly(macro[k])
        if MACRO_VARS[k].kind == "rate":
            out[k] = q.diff()
        else:
            out[k] = pct(q, 4).diff() if basis == "momentum" else pct(q)
    return pd.DataFrame(out)


def classify(signal: pd.Series, deadband: float = 0.0) -> pd.Series:
    """Label each period Up / Down, or Flat when |signal| <= deadband."""
    labels = np.where(signal > deadband, "Up", np.where(signal < -deadband, "Down", "Flat"))
    return pd.Series(labels, index=signal.index, dtype=object).where(signal.notna())


def policy_cycle(moves: pd.Series, max_pause: int = 4) -> pd.Series:
    """Turn per-quarter Up/Flat/Down policy moves into hiking / cutting cycles.

    Flat quarters sitting between two moves in the same direction, no more than max_pause
    quarters apart, join that cycle (a pause mid-cycle is still a hiking or cutting cycle).
    Flat quarters between moves in opposite directions, or after the latest move, stay Flat.
    """
    r = moves.dropna()
    vals = r.to_numpy(dtype=object).copy()
    move_pos = [i for i, v in enumerate(vals) if v != "Flat"]
    for a, b in zip(move_pos, move_pos[1:], strict=False):
        if vals[a] == vals[b] and 0 < b - a - 1 <= max_pause:
            vals[a + 1 : b] = vals[a]
    out = moves.copy().astype(object)
    out.loc[r.index] = vals
    return out


def fed_regime(target: pd.Series, cycle: bool = True, max_pause: int = 4) -> pd.Series:
    """Quarterly Fed regime from the policy target: Up = hiking, Down = cutting, Flat = on hold."""
    # Target moves come in 25bp steps; the 1bp band only absorbs float noise.
    moves = classify(to_quarterly(target).diff(), deadband=0.01)
    return policy_cycle(moves, max_pause) if cycle else moves


def _annualised(r: pd.Series) -> float:
    """Compound quarterly % returns and express them as an annual rate."""
    return ((1 + r / 100).prod() ** (4 / len(r)) - 1) * 100


def regime_stats(ret: pd.Series, regimes: pd.DataFrame) -> pd.DataFrame:
    """Dollar (or FX) return statistics for each variable's Up/Flat/Down regime."""
    rows = []
    for var in regimes.columns:
        df = pd.concat({"ret": ret, "regime": regimes[var]}, axis=1).dropna()
        for reg in REGIME_ORDER:
            r = df.loc[df["regime"] == reg, "ret"].astype(float)
            if r.empty:
                continue
            rows.append(
                {
                    "variable": var,
                    "regime": reg,
                    "mean": r.mean(),
                    "median": r.median(),
                    "hit_rate": (r > 0).mean() * 100,
                    "n": len(r),
                    "ann": _annualised(r),
                    "std": r.std(),
                }
            )
    cols = ["variable", "regime", "mean", "median", "hit_rate", "n", "ann", "std"]
    return pd.DataFrame(rows, columns=cols)


def episode_returns(returns: dict[str, pd.Series], regime: pd.Series) -> pd.DataFrame:
    """Cumulative % move of each series over every contiguous run of one regime."""
    rows = []
    for start, end, reg in regime_blocks(regime):
        row = {"start": start, "end": end, "regime": reg}
        for name, ret in returns.items():
            r = ret[(ret.index >= start) & (ret.index <= end)].dropna()
            # Only report an episode for a series that covers all of it.
            n_q = int(((regime.index >= start) & (regime.index <= end)).sum())
            row[name] = ((1 + r / 100).prod() - 1) * 100 if len(r) == n_q else np.nan
        row["quarters"] = n_q
        rows.append(row)
    return pd.DataFrame(rows)


def joint_regime_stats(ret: pd.Series, a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Return statistics for every combination of two regime series (e.g. US vs Japan)."""
    df = pd.concat({"ret": ret, "a": a, "b": b}, axis=1).dropna()
    rows = []
    for ra in REGIME_ORDER:
        for rb in REGIME_ORDER:
            r = df.loc[(df["a"] == ra) & (df["b"] == rb), "ret"].astype(float)
            if r.empty:
                continue
            rows.append(
                {
                    "a": ra,
                    "b": rb,
                    "mean": r.mean(),
                    "ann": _annualised(r),
                    "hit_rate": (r > 0).mean() * 100,
                    "n": len(r),
                }
            )
    return pd.DataFrame(rows, columns=["a", "b", "mean", "ann", "hit_rate", "n"])


def pair_regime_stats(returns: dict[str, pd.Series], regimes: pd.DataFrame) -> pd.DataFrame:
    """regime_stats for several return series, stacked with a 'pair' column."""
    frames = [regime_stats(r, regimes).assign(pair=name) for name, r in returns.items()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def regime_matrix(
    ret: pd.Series, regimes: pd.DataFrame, a: str, b: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean return and count for each combination of two variables' regimes (rows=a, cols=b)."""
    df = pd.concat({"ret": ret, "a": regimes[a], "b": regimes[b]}, axis=1).dropna()
    df["ret"] = df["ret"].astype(float)
    g = df.groupby(["a", "b"])["ret"]
    mean = g.mean().unstack()
    count = g.size().unstack()
    rows = [r for r in REGIME_ORDER if r in mean.index]
    cols = [c for c in REGIME_ORDER if c in mean.columns]
    return mean.reindex(index=rows, columns=cols), count.reindex(index=rows, columns=cols).fillna(
        0
    ).astype(int)


def regime_blocks(regime: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Contiguous runs of the same regime as (start, end, label) spans covering whole quarters."""
    r = regime.dropna()
    if r.empty:
        return []
    run_id = (r != r.shift()).cumsum()
    blocks = []
    for _, chunk in r.groupby(run_id):
        start = chunk.index[0].to_period("Q").start_time
        end = chunk.index[-1].to_period("Q").end_time
        blocks.append((start, end, chunk.iloc[0]))
    return blocks


@dataclass(frozen=True)
class Sensitivity:
    beta: pd.Series  # % FX move per unit of each driver
    se: pd.Series
    cov: pd.DataFrame
    r2: float
    nobs: int

    @property
    def tstat(self) -> pd.Series:
        return self.beta / self.se


def fit_sensitivity(ret: pd.Series, drivers: pd.DataFrame, min_obs: int = 12) -> Sensitivity | None:
    """OLS of quarterly FX % change on contemporaneous macro changes, with robust (HC1) errors."""
    df = pd.concat([ret.rename("_y"), drivers], axis=1).dropna()
    if len(df) < min_obs:
        return None
    cols = list(drivers.columns)
    res = sm.OLS(df["_y"], sm.add_constant(df[cols])).fit(cov_type="HC1")
    return Sensitivity(
        beta=res.params[cols],
        se=res.bse[cols],
        cov=res.cov_params().loc[cols, cols],
        r2=float(res.rsquared),
        nobs=int(res.nobs),
    )


def scenario_impact(sens: Sensitivity, shocks: dict[str, float]) -> tuple[pd.Series, float, float]:
    """Per-driver contribution, total % move and its standard error for a set of shocks."""
    s = pd.Series(shocks, dtype=float).reindex(sens.beta.index).fillna(0.0)
    contrib = sens.beta * s
    se = float(np.sqrt(s.to_numpy() @ sens.cov.to_numpy() @ s.to_numpy()))
    return contrib, float(contrib.sum()), se


def trailing_returns(s: pd.Series) -> dict[str, float]:
    """% change over 1M, 3M, YTD and 1Y to the latest observation."""
    s = s.dropna()
    last_date, last = s.index[-1], s.iloc[-1]
    bases = {
        "1M": s.asof(last_date - pd.DateOffset(months=1)),
        "3M": s.asof(last_date - pd.DateOffset(months=3)),
        "YTD": s.asof(pd.Timestamp(last_date.year - 1, 12, 31)),
        "1Y": s.asof(last_date - pd.DateOffset(years=1)),
    }
    return {k: (last / b - 1) * 100 if pd.notna(b) else np.nan for k, b in bases.items()}


def rebase(s: pd.Series, start) -> pd.Series:
    s = s[s.index >= pd.Timestamp(start)].dropna()
    return s / s.iloc[0] * 100 if not s.empty else s
