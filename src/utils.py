"""
utils.py
========
Shared utility functions used across the research framework.

All functions are pure (no side-effects) and operate on pandas/numpy objects.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cross_sectional_zscore(
    series: pd.Series,
    winsorize_std: float = 3.0,
) -> pd.Series:
    """Compute a cross-sectional z-score, optionally winsorized.

    Args:
        series: A pandas Series of raw factor values indexed by ticker.
        winsorize_std: Clip values more than this many standard deviations
            from the mean before scoring.  Set to ``np.inf`` to disable.

    Returns:
        A pandas Series of z-scores with the same index as ``series``,
        with NaN preserved for tickers that had NaN input.
    """
    s = series.copy().astype(float)
    valid = s.dropna()
    if len(valid) < 3:
        return pd.Series(np.nan, index=series.index)

    mu = valid.mean()
    sigma = valid.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(0.0, index=series.index)

    z = (s - mu) / sigma
    if winsorize_std < np.inf:
        z = z.clip(lower=-winsorize_std, upper=winsorize_std)
    return z


def winsorize_series(
    series: pd.Series,
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.Series:
    """Winsorize a series to the [lower, upper] quantile range.

    Args:
        series: Input pandas Series.
        lower: Lower quantile clip bound (e.g. 0.05 = 5th percentile).
        upper: Upper quantile clip bound (e.g. 0.95 = 95th percentile).

    Returns:
        Winsorized pandas Series.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def percentile_rank(series: pd.Series) -> pd.Series:
    """Rank values as cross-sectional percentile (0–1).

    Ties are broken by average rank.  NaN values are excluded and
    returned as NaN.

    Args:
        series: Raw factor values indexed by ticker.

    Returns:
        Percentile ranks between 0 and 1.
    """
    return series.rank(pct=True, na_option="keep")


def information_coefficient(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """Compute the Information Coefficient between factor scores and returns.

    Args:
        factor_scores: Cross-sectional factor scores indexed by ticker.
        forward_returns: Forward return for the same set of tickers.
        method: ``"spearman"`` (default, rank IC) or ``"pearson"`` (linear IC).

    Returns:
        Scalar IC value.  NaN if insufficient data.
    """
    # Align and drop NaN pairs
    df = pd.concat(
        [factor_scores.rename("factor"), forward_returns.rename("ret")],
        axis=1,
    ).dropna()
    if len(df) < 5:
        return np.nan

    if method == "spearman":
        ic, _ = stats.spearmanr(df["factor"], df["ret"])
    else:
        ic, _ = stats.pearsonr(df["factor"], df["ret"])
    return float(ic)


# ---------------------------------------------------------------------------
# Date / rebalance helpers
# ---------------------------------------------------------------------------

def get_rebalance_dates(
    start: str,
    end: str,
    freq: str = "M",
) -> pd.DatetimeIndex:
    """Return business-day-adjusted month-end (or other frequency) rebalance dates.

    Args:
        start: Start date string (``"YYYY-MM-DD"``).
        end: End date string (``"YYYY-MM-DD"``).
        freq: Pandas offset alias.  ``"M"`` = month end, ``"Q"`` = quarter end.

    Returns:
        DatetimeIndex of rebalance dates.
    """
    dates = pd.date_range(start=start, end=end, freq=freq)
    # Shift any weekends / holidays to the previous business day
    bday = pd.offsets.BDay()
    adjusted = []
    for d in dates:
        if d.weekday() >= 5:  # Saturday=5, Sunday=6
            d = d - bday
        adjusted.append(d)
    return pd.DatetimeIndex(adjusted)


def lag_fundamentals(df: pd.DataFrame, lag_days: int = 45) -> pd.DataFrame:
    """Shift a fundamentals DataFrame forward by ``lag_days`` business days.

    This prevents look-ahead bias: a quarterly filing for period ending
    March 31 is not available until ~May 15 (≈ 45 calendar days later).

    Args:
        df: DataFrame indexed by date (or MultiIndex with date as one level).
        lag_days: Number of *calendar* days to lag.

    Returns:
        DataFrame with date index shifted forward by ``lag_days`` days.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = df.index + pd.Timedelta(days=lag_days)
    return df


# ---------------------------------------------------------------------------
# Return calculations
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame.

    Args:
        prices: DataFrame of adjusted close prices (dates × tickers).

    Returns:
        DataFrame of daily log returns, same shape, first row NaN.
    """
    return np.log(prices / prices.shift(1))


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple (arithmetic) returns from a price DataFrame.

    Args:
        prices: DataFrame of adjusted close prices (dates × tickers).

    Returns:
        DataFrame of daily simple returns.
    """
    return prices.pct_change()


def annualize_return(cum_return: float, n_years: float) -> float:
    """Convert a cumulative return to a compound annual growth rate (CAGR).

    Args:
        cum_return: Total cumulative return as a decimal (e.g. 1.5 = 150%).
        n_years: Number of years in the sample.

    Returns:
        CAGR as a decimal.
    """
    if n_years <= 0:
        return np.nan
    return (1.0 + cum_return) ** (1.0 / n_years) - 1.0


def annualize_vol(daily_vol: float, trading_days: int = 252) -> float:
    """Annualize a daily standard deviation.

    Args:
        daily_vol: Daily standard deviation.
        trading_days: Number of trading days per year (default 252).

    Returns:
        Annualized volatility.
    """
    return daily_vol * np.sqrt(trading_days)


def sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.02,
    trading_days: int = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Daily return series.
        rf_annual: Annual risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Sharpe ratio (annualized).
    """
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    excess = returns - rf_daily
    if excess.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(trading_days) * excess.mean() / excess.std(ddof=1))


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from a return series.

    Args:
        returns: Daily simple returns.

    Returns:
        Maximum drawdown as a *negative* decimal.
    """
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    drawdown = cum / roll_max - 1
    return float(drawdown.min())


def calmar_ratio(
    returns: pd.Series,
    rf_annual: float = 0.02,
    trading_days: int = 252,
) -> float:
    """Compute the Calmar ratio (CAGR / |max drawdown|).

    Args:
        returns: Daily return series.
        rf_annual: Annual risk-free rate (not subtracted here; convention varies).
        trading_days: Trading days per year.

    Returns:
        Calmar ratio.
    """
    n_years = len(returns) / trading_days
    cum_ret = (1 + returns).prod() - 1
    cagr = annualize_return(cum_ret, n_years)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return float(cagr / abs(mdd))


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def equal_weight(tickers: Sequence[str]) -> dict[str, float]:
    """Return equal-weight portfolio as a weight dict.

    Args:
        tickers: List of ticker strings.

    Returns:
        Dict mapping ticker → weight (all equal, summing to 1).
    """
    n = len(tickers)
    if n == 0:
        return {}
    w = 1.0 / n
    return {t: w for t in tickers}


def dollar_neutral_weights(
    long_tickers: Sequence[str],
    short_tickers: Sequence[str],
) -> dict[str, float]:
    """Construct a dollar-neutral long-short weight dict.

    Long side: equal-weight +0.5 each; short side: equal-weight -0.5 each.

    Args:
        long_tickers: Tickers to hold long.
        short_tickers: Tickers to hold short.

    Returns:
        Dict mapping ticker → signed weight.  Gross exposure = 1.
    """
    weights: dict[str, float] = {}
    n_long = len(long_tickers)
    n_short = len(short_tickers)
    if n_long > 0:
        for t in long_tickers:
            weights[t] = 0.5 / n_long
    if n_short > 0:
        for t in short_tickers:
            weights[t] = -0.5 / n_short
    return weights


def portfolio_return(
    weights: dict[str, float],
    returns: pd.Series,
) -> float:
    """Compute a single-period portfolio return.

    Args:
        weights: Dict of ticker → weight.
        returns: Series of ticker returns for the period.

    Returns:
        Scalar portfolio return.
    """
    ret = 0.0
    for ticker, w in weights.items():
        if ticker in returns.index and not np.isnan(returns[ticker]):
            ret += w * returns[ticker]
    return ret


def compute_turnover(
    weights_t: dict[str, float],
    weights_t1: dict[str, float],
) -> float:
    """Compute one-way portfolio turnover between two rebalances.

    Args:
        weights_t: Previous period weights.
        weights_t1: New period weights.

    Returns:
        One-way turnover as a fraction of portfolio (0–1).
    """
    all_tickers = set(weights_t) | set(weights_t1)
    turnover = 0.0
    for t in all_tickers:
        w_old = weights_t.get(t, 0.0)
        w_new = weights_t1.get(t, 0.0)
        turnover += abs(w_new - w_old)
    return turnover / 2.0  # one-way


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_pct(x: float, decimals: int = 1) -> str:
    """Format a decimal as a percentage string.

    Args:
        x: Decimal value (e.g. 0.1523 → ``"15.2%"``).
        decimals: Number of decimal places.

    Returns:
        Formatted string.
    """
    if np.isnan(x):
        return "N/A"
    return f"{x * 100:.{decimals}f}%"


def format_number(x: float, decimals: int = 2) -> str:
    """Format a float with commas and specified decimal places.

    Args:
        x: Float to format.
        decimals: Decimal places.

    Returns:
        Formatted string.
    """
    if np.isnan(x):
        return "N/A"
    return f"{x:,.{decimals}f}"


def suppress_warnings() -> None:
    """Suppress common data-download and deprecation warnings for notebook runs."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")
