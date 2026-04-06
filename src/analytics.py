"""
analytics.py
============
Backtesting, performance attribution, and regime analysis engine.

Implements:
  - ``backtest_portfolio``        — Daily P&L, cumulative returns, turnover
  - ``performance_summary``       — Full statistics table
  - ``relative_performance``      — Active return, tracking error, IR
  - ``factor_ic_panel``           — Time-series IC for each factor
  - ``quintile_analysis``         — Quintile portfolio returns for a factor
  - ``regime_conditional``        — Conditional performance by macro regime
  - ``rolling_performance``       — Rolling Sharpe and drawdown
  - ``drawdown_series``           — Full drawdown time series
  - ``monthly_return_matrix``     — Year × month return heatmap data
  - ``fama_french_regression``    — 5-factor alpha/beta decomposition
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils import (
    sharpe_ratio, max_drawdown, calmar_ratio, annualize_return,
    annualize_vol, information_coefficient, compute_turnover,
    portfolio_return,
)


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def backtest_portfolio(
    weights_history: dict[pd.Timestamp, dict[str, float]],
    prices: pd.DataFrame,
    cost_bps: float = 15.0,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Simulate daily portfolio returns from a weights history.

    At each rebalance date, the portfolio is rebalanced to the new target
    weights.  Transaction costs are applied to the one-way turnover.
    Between rebalance dates, holdings drift with prices (no intra-period
    rebalancing).

    Args:
        weights_history: Dict mapping rebalance date → weight dict.
            Produced by ``portfolio.build_weights_history``.
        prices: Adjusted close price DataFrame (dates × tickers).
        cost_bps: One-way transaction cost in basis points (default: 15).
        start: Optional start date for the backtest window.
        end: Optional end date for the backtest window.

    Returns:
        DataFrame indexed by date with columns:
          - ``portfolio_return``: Daily portfolio simple return (net of costs).
          - ``gross_return``: Daily portfolio return before costs.
          - ``turnover``: One-way turnover on rebalance days (0 otherwise).
          - ``cost_drag``: Transaction cost drag on rebalance days.
          - ``cumulative_return``: Cumulative gross-of-cost return.
          - ``cumulative_net_return``: Cumulative net-of-cost return.
    """
    # Restrict price series to backtest window
    if start:
        prices = prices.loc[start:]
    if end:
        prices = prices.loc[:end]

    daily_returns = prices.pct_change().fillna(0.0)
    rebalance_dates = sorted(weights_history.keys())

    results: list[dict] = []
    current_weights: dict[str, float] = {}

    for i, date in enumerate(prices.index):
        # Check if this is a rebalance date
        is_rebalance = date in weights_history

        # Compute today's stock returns
        day_rets = daily_returns.loc[date]

        # Pre-rebalance: drift current holdings with today's returns
        gross_ret = portfolio_return(current_weights, day_rets)

        # Update drifted weights (buy-and-hold drift between rebalances)
        if current_weights:
            drifted: dict[str, float] = {}
            total_val = sum(
                w * (1.0 + day_rets.get(t, 0.0))
                for t, w in current_weights.items()
                if not np.isnan(day_rets.get(t, 0.0))
            )
            if total_val > 0:
                for t, w in current_weights.items():
                    r = day_rets.get(t, 0.0)
                    if not np.isnan(r):
                        drifted[t] = w * (1.0 + r) / total_val
            current_weights = drifted

        # Rebalance: compute turnover and cost, then switch to new weights
        turnover = 0.0
        cost_drag = 0.0
        if is_rebalance:
            new_weights = weights_history[date]
            turnover = compute_turnover(current_weights, new_weights)
            cost_drag = turnover * cost_bps / 10_000.0
            current_weights = new_weights.copy()

        net_ret = gross_ret - cost_drag

        results.append({
            "date": date,
            "gross_return": gross_ret,
            "portfolio_return": net_ret,
            "turnover": turnover,
            "cost_drag": cost_drag,
        })

    df = pd.DataFrame(results).set_index("date")
    df["cumulative_gross_return"] = (1 + df["gross_return"]).cumprod() - 1
    df["cumulative_net_return"]   = (1 + df["portfolio_return"]).cumprod() - 1
    return df


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def performance_summary(
    returns: pd.Series,
    rf: float = 0.02,
    trading_days: int = 252,
    label: str = "Strategy",
) -> pd.Series:
    """Compute a comprehensive performance summary for a return series.

    Args:
        returns: Daily simple return series.
        rf: Annual risk-free rate.
        trading_days: Trading days per year.
        label: Display label for the summary.

    Returns:
        pd.Series with named statistics:
          total_return, cagr, annualized_vol, sharpe_ratio,
          max_drawdown, calmar_ratio, sortino_ratio,
          hit_rate (fraction of positive return days),
          skewness, kurtosis, var_95 (5th pct daily return).
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return pd.Series(name=label, dtype=float)

    total_ret = float((1 + returns).prod() - 1)
    n_years = len(returns) / trading_days
    cagr = annualize_return(total_ret, n_years)
    ann_vol = annualize_vol(returns.std(ddof=1), trading_days)
    sr = sharpe_ratio(returns, rf, trading_days)
    mdd = max_drawdown(returns)
    calmar = calmar_ratio(returns, rf, trading_days)

    # Sortino: downside deviation (below risk-free)
    rf_daily = (1 + rf) ** (1 / trading_days) - 1
    downside = returns[returns < rf_daily] - rf_daily
    dd_dev = annualize_vol(downside.std(ddof=1), trading_days) if len(downside) > 1 else np.nan
    sortino = (cagr - rf) / dd_dev if (dd_dev and dd_dev > 0) else np.nan

    hit_rate = float((returns > 0).mean())
    skew = float(scipy_stats.skew(returns))
    kurt = float(scipy_stats.kurtosis(returns))
    var95 = float(returns.quantile(0.05))

    # Average monthly return
    monthly = (1 + returns).resample("M").prod() - 1
    avg_monthly = float(monthly.mean())
    best_month = float(monthly.max())
    worst_month = float(monthly.min())

    return pd.Series({
        "Total Return":        total_ret,
        "CAGR":                cagr,
        "Ann. Volatility":     ann_vol,
        "Sharpe Ratio":        sr,
        "Sortino Ratio":       sortino,
        "Max Drawdown":        mdd,
        "Calmar Ratio":        calmar,
        "Hit Rate":            hit_rate,
        "Skewness":            skew,
        "Excess Kurtosis":     kurt,
        "VaR (95%, daily)":    var95,
        "Avg Monthly Return":  avg_monthly,
        "Best Month":          best_month,
        "Worst Month":         worst_month,
    }, name=label)


def relative_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0.02,
    trading_days: int = 252,
    label: str = "Strategy vs. Benchmark",
) -> pd.Series:
    """Compute relative (active) performance statistics.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns (e.g., SMH).
        rf: Annual risk-free rate.
        trading_days: Trading days per year.
        label: Display label.

    Returns:
        pd.Series with: active_return, tracking_error, information_ratio,
        beta, alpha, correlation, up_capture, down_capture.
    """
    # Align
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    p = portfolio_returns.loc[common].dropna()
    b = benchmark_returns.loc[common].dropna()

    common2 = p.index.intersection(b.index)
    p = p.loc[common2]
    b = b.loc[common2]

    if len(p) < 10:
        return pd.Series(name=label, dtype=float)

    active = p - b
    ann_active = active.mean() * trading_days
    te = active.std(ddof=1) * np.sqrt(trading_days)
    ir = ann_active / te if te > 0 else np.nan

    # Beta / alpha via OLS
    rf_daily = (1 + rf) ** (1 / trading_days) - 1
    p_excess = p - rf_daily
    b_excess = b - rf_daily
    slope, intercept, r_val, p_val, se = scipy_stats.linregress(b_excess, p_excess)
    beta = float(slope)
    alpha_daily = float(intercept)
    alpha_ann = alpha_daily * trading_days

    # Up/down capture
    up_mask = b > 0
    down_mask = b < 0

    up_cap = (
        ((1 + p[up_mask]).prod() ** (trading_days / max(up_mask.sum(), 1)) - 1)
        / ((1 + b[up_mask]).prod() ** (trading_days / max(up_mask.sum(), 1)) - 1)
        if up_mask.sum() > 0 and (1 + b[up_mask]).prod() > 1 else np.nan
    )
    down_cap = (
        ((1 + p[down_mask]).prod() ** (trading_days / max(down_mask.sum(), 1)) - 1)
        / ((1 + b[down_mask]).prod() ** (trading_days / max(down_mask.sum(), 1)) - 1)
        if down_mask.sum() > 0 and (1 + b[down_mask]).prod() < 1 else np.nan
    )

    return pd.Series({
        "Active Return (Ann.)":     ann_active,
        "Tracking Error (Ann.)":    te,
        "Information Ratio":        ir,
        "Beta":                     beta,
        "Alpha (Ann.)":             alpha_ann,
        "Correlation":              float(p.corr(b)),
        "Up-Market Capture":        up_cap,
        "Down-Market Capture":      down_cap,
    }, name=label)


# ---------------------------------------------------------------------------
# IC analysis
# ---------------------------------------------------------------------------

def factor_ic_panel(
    factor_panel: dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    forward_periods: int = 21,  # ~1 month
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute monthly Information Coefficients for each factor.

    For each rebalance date, computes the IC between each factor's z-score
    and the forward ``forward_periods``-day return.

    Args:
        factor_panel: Dict mapping rebalance date → factor DataFrame.
        prices: Adjusted close prices (dates × tickers).
        forward_periods: Number of trading days for the forward return
            window (default: 21 ≈ 1 month).
        method: IC method: ``"spearman"`` (rank IC) or ``"pearson"``.

    Returns:
        DataFrame indexed by rebalance date with one column per factor.
        Values are IC scores (range approximately −1 to +1).
    """
    records: list[dict] = []
    rebalance_dates = sorted(factor_panel.keys())

    for date in rebalance_dates:
        factor_df = factor_panel.get(date)
        if factor_df is None or factor_df.empty:
            continue

        # Compute forward returns
        price_dates = prices.index
        start_idx = price_dates[price_dates >= date]
        if len(start_idx) < forward_periods + 1:
            continue
        start_price_date = start_idx[0]
        end_price_date_idx = price_dates[price_dates >= date]
        if len(end_price_date_idx) <= forward_periods:
            continue
        end_price_date = end_price_date_idx[min(forward_periods, len(end_price_date_idx) - 1)]

        fwd_returns = (
            prices.loc[end_price_date] / prices.loc[start_price_date] - 1.0
        )

        row: dict = {"date": date}
        for factor_col in factor_df.columns:
            ic = information_coefficient(
                factor_df[factor_col],
                fwd_returns,
                method=method,
            )
            row[factor_col] = ic

        records.append(row)

    if not records:
        return pd.DataFrame()

    ic_df = pd.DataFrame(records).set_index("date")
    return ic_df


def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize IC statistics for each factor.

    Args:
        ic_df: Output of ``factor_ic_panel``.

    Returns:
        DataFrame indexed by factor name with columns:
          IC Mean, IC Std, IC IR (mean/std), t-Stat, Hit Rate.
    """
    records = []
    for col in ic_df.columns:
        series = ic_df[col].dropna()
        if len(series) == 0:
            continue
        mean_ic = series.mean()
        std_ic  = series.std(ddof=1)
        ic_ir   = mean_ic / std_ic if std_ic > 0 else np.nan
        t_stat  = mean_ic / (std_ic / np.sqrt(len(series))) if std_ic > 0 else np.nan
        hit_rate = (series > 0).mean()
        records.append({
            "Factor": col,
            "IC Mean": mean_ic,
            "IC Std": std_ic,
            "IC IR": ic_ir,
            "t-Stat": t_stat,
            "Hit Rate": hit_rate,
            "N Obs": len(series),
        })

    return pd.DataFrame(records).set_index("Factor")


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------

def quintile_analysis(
    factor_panel: dict[pd.Timestamp, pd.DataFrame],
    factor_name: str,
    prices: pd.DataFrame,
    n_quantiles: int = 5,
    forward_periods: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute quintile average returns and cumulative long-short for a factor.

    At each rebalance date, tickers are sorted into ``n_quantiles`` buckets
    by the given factor.  The equal-weight return of each bucket over the
    forward ``forward_periods`` days is recorded.

    Args:
        factor_panel: Dict of factor DataFrames.
        factor_name: Column name of the factor to analyze.
        prices: Adjusted close prices.
        n_quantiles: Number of quantile buckets (default: 5).
        forward_periods: Forward holding period in trading days.

    Returns:
        Tuple of:
          - avg_returns: DataFrame (n_quantiles × dates) of average returns.
          - ls_returns: Series of long-short (Q5 − Q1) returns over time.
    """
    quantile_returns: dict[int, list[float]] = {q: [] for q in range(1, n_quantiles + 1)}
    dates_list: list[pd.Timestamp] = []
    rebalance_dates = sorted(factor_panel.keys())

    for date in rebalance_dates:
        factor_df = factor_panel.get(date)
        if factor_df is None or factor_df.empty or factor_name not in factor_df.columns:
            continue

        factor_scores = factor_df[factor_name].dropna()
        if len(factor_scores) < n_quantiles:
            continue

        # Compute forward returns
        price_dates = prices.index
        start_idx = price_dates[price_dates >= date]
        if len(start_idx) <= forward_periods:
            continue
        start_date = start_idx[0]
        end_date   = start_idx[min(forward_periods, len(start_idx) - 1)]

        fwd_returns = prices.loc[end_date] / prices.loc[start_date] - 1.0

        # Assign quantiles
        labels = pd.qcut(factor_scores, q=n_quantiles, labels=False, duplicates="drop")
        labels += 1  # 1-indexed

        dates_list.append(date)
        for q in range(1, n_quantiles + 1):
            q_tickers = labels[labels == q].index
            q_rets = fwd_returns.reindex(q_tickers).dropna()
            if len(q_rets) > 0:
                quantile_returns[q].append(float(q_rets.mean()))
            else:
                quantile_returns[q].append(np.nan)

    if not dates_list:
        return pd.DataFrame(), pd.Series(dtype=float)

    avg_df = pd.DataFrame(
        {f"Q{q}": quantile_returns[q] for q in range(1, n_quantiles + 1)},
        index=dates_list,
    )

    ls = avg_df.get(f"Q{n_quantiles}", pd.Series(dtype=float)) - avg_df.get("Q1", pd.Series(dtype=float))
    ls_cum = (1 + ls.dropna()).cumprod() - 1

    return avg_df, ls_cum


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def regime_conditional(
    returns: pd.Series,
    macro_series: pd.Series,
    threshold: str = "median",
    label: str = "Strategy",
    trading_days: int = 252,
) -> pd.DataFrame:
    """Split returns by macro regime and compute conditional performance.

    Splits the macro series at the median (or another threshold) into
    "high" and "low" regimes, then computes performance statistics in each.

    Args:
        returns: Daily portfolio returns.
        macro_series: Daily or monthly macro variable (e.g., T10Y2Y).
            Will be forward-filled to daily frequency.
        threshold: ``"median"`` (default) splits at the time-series median.
            Can also be a float (absolute threshold value).
        label: Strategy label for display.
        trading_days: Trading days per year.

    Returns:
        DataFrame with rows = [high_regime, low_regime] and
        columns = [CAGR, Volatility, Sharpe, Max Drawdown, N Days].
    """
    # Align macro to returns frequency (forward-fill)
    macro_daily = macro_series.reindex(returns.index, method="ffill").dropna()
    common = returns.index.intersection(macro_daily.index)
    ret_aligned = returns.loc[common]
    macro_aligned = macro_daily.loc[common]

    if isinstance(threshold, str) and threshold == "median":
        split = float(macro_aligned.median())
    else:
        split = float(threshold)

    high_mask = macro_aligned >= split
    low_mask  = macro_aligned < split

    rows = []
    for mask, regime_name in [(high_mask, f"High {macro_series.name}"),
                               (low_mask,  f"Low {macro_series.name}")]:
        r = ret_aligned[mask].dropna()
        if len(r) < 20:
            rows.append({
                "Regime": regime_name,
                "CAGR": np.nan, "Volatility": np.nan,
                "Sharpe": np.nan, "Max Drawdown": np.nan,
                "N Days": len(r),
            })
            continue

        n_years = len(r) / trading_days
        total_ret = float((1 + r).prod() - 1)
        cagr = annualize_return(total_ret, n_years)
        vol  = annualize_vol(r.std(ddof=1), trading_days)
        sr   = sharpe_ratio(r, trading_days=trading_days)
        mdd  = max_drawdown(r)

        rows.append({
            "Regime": regime_name,
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sr,
            "Max Drawdown": mdd,
            "N Days": int(mask.sum()),
        })

    return pd.DataFrame(rows).set_index("Regime")


# ---------------------------------------------------------------------------
# Auxiliary analytics
# ---------------------------------------------------------------------------

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the full drawdown time series (peak-to-trough drawdown at each date).

    Args:
        returns: Daily simple return series.

    Returns:
        Series of drawdown values (0 at peaks, negative at troughs).
    """
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1
    dd.name = "drawdown"
    return dd


def monthly_return_matrix(returns: pd.Series) -> pd.DataFrame:
    """Reshape daily returns into a year × month matrix of monthly returns.

    Args:
        returns: Daily simple returns.

    Returns:
        DataFrame with years as index, months (1–12) as columns.
        Cell values are monthly compounded returns (e.g. 0.05 = 5%).
    """
    monthly = (1 + returns).resample("M").prod() - 1
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    matrix = df.pivot(index="year", columns="month", values="ret")
    matrix.columns.name = None
    matrix.index.name   = None
    return matrix


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    rf: float = 0.02,
    trading_days: int = 252,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Args:
        returns: Daily return series.
        window: Rolling window size in trading days.
        rf: Annual risk-free rate.
        trading_days: Trading days per year.

    Returns:
        Series of rolling Sharpe ratios.
    """
    rf_daily = (1 + rf) ** (1 / trading_days) - 1
    excess = returns - rf_daily
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std(ddof=1)
    result = roll_mean / roll_std * np.sqrt(trading_days)
    result.name = "rolling_sharpe"
    return result


def fama_french_regression(
    portfolio_returns: pd.Series,
    ff5_factors: pd.DataFrame,
    trading_days: int = 252,
) -> dict:
    """Run OLS regression of portfolio excess returns on FF5 factors.

    Decomposes portfolio returns into:
      Alpha (Jensen's alpha), Mkt-RF, SMB, HML, RMW, CMA exposures.

    Args:
        portfolio_returns: Daily portfolio return series.
        ff5_factors: DataFrame with columns [Mkt-RF, SMB, HML, RMW, CMA, RF].
            Typically from Ken French's data library (already in decimal form).
        trading_days: For alpha annualization.

    Returns:
        Dict with keys: alpha_daily, alpha_annual, betas (dict),
        r_squared, t_stats (dict), residual_std.
    """
    common = portfolio_returns.index.intersection(ff5_factors.index)
    p = portfolio_returns.loc[common]
    ff = ff5_factors.loc[common]

    # Excess returns
    rf = ff["RF"] if "RF" in ff.columns else pd.Series(0.0, index=common)
    excess_p = p - rf

    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if c in ff.columns]
    if not factor_cols:
        return {}

    X = ff[factor_cols].copy()
    X = X.loc[common]

    # Add intercept
    X_const = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)

    # Align
    aligned = pd.concat([excess_p.rename("excess_ret"), X_const], axis=1).dropna()
    y = aligned["excess_ret"]
    X_reg = aligned[[c for c in X_const.columns]]

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(fit_intercept=False).fit(X_reg.values, y.values)
    coeffs = dict(zip(X_reg.columns, reg.coef_))

    alpha_daily = coeffs.get("const", np.nan)
    alpha_annual = alpha_daily * trading_days

    residuals = y.values - reg.predict(X_reg.values)
    residual_std = float(np.std(residuals, ddof=len(X_reg.columns)))
    r2 = float(reg.score(X_reg.values, y.values))

    # t-stats
    n = len(aligned)
    k = len(X_reg.columns)
    mse = np.sum(residuals ** 2) / (n - k)
    try:
        cov_params = mse * np.linalg.inv(X_reg.values.T @ X_reg.values)
        se = np.sqrt(np.diag(cov_params))
        t_stats = dict(zip(X_reg.columns, reg.coef_ / se))
    except np.linalg.LinAlgError:
        t_stats = {c: np.nan for c in X_reg.columns}

    betas = {k: v for k, v in coeffs.items() if k != "const"}

    return {
        "alpha_daily":   alpha_daily,
        "alpha_annual":  alpha_annual,
        "betas":         betas,
        "r_squared":     r2,
        "t_stats":       t_stats,
        "residual_std":  residual_std,
        "n_obs":         n,
    }


def subperiod_analysis(
    weights_history: dict[pd.Timestamp, dict[str, float]],
    prices: pd.DataFrame,
    benchmark_returns: pd.Series,
    subperiods: list[tuple[str, str]],
    cost_bps: float = 15.0,
    rf: float = 0.02,
) -> pd.DataFrame:
    """Compute performance statistics for each defined sub-period.

    Args:
        weights_history: Portfolio weights history dict.
        prices: Price DataFrame.
        benchmark_returns: Benchmark return series (for IR calculation).
        subperiods: List of (start, end) date string tuples.
        cost_bps: Transaction cost in basis points.
        rf: Annual risk-free rate.

    Returns:
        DataFrame indexed by sub-period label, with performance stats.
    """
    results = []
    for start, end in subperiods:
        label = f"{start[:4]}–{end[:4]}"
        bt = backtest_portfolio(weights_history, prices, cost_bps, start=start, end=end)
        port_ret = bt["portfolio_return"]
        stats = performance_summary(port_ret, rf=rf, label=label)

        bm = benchmark_returns.loc[start:end]
        rel = relative_performance(port_ret, bm, rf=rf, label=label)

        combined = pd.concat([stats, rel])
        combined.name = label
        results.append(combined)

    return pd.DataFrame(results)


def transaction_cost_sensitivity(
    weights_history: dict[pd.Timestamp, dict[str, float]],
    prices: pd.DataFrame,
    cost_levels: list[float],
    rf: float = 0.02,
) -> pd.DataFrame:
    """Run the backtest at multiple transaction cost levels.

    Args:
        weights_history: Portfolio weights.
        prices: Price DataFrame.
        cost_levels: List of transaction costs in bps (e.g. [0, 15, 30]).
        rf: Annual risk-free rate.

    Returns:
        DataFrame with performance stats for each cost level.
    """
    results = []
    for bps in cost_levels:
        bt = backtest_portfolio(weights_history, prices, cost_bps=bps)
        port_ret = bt["portfolio_return"]
        stats = performance_summary(port_ret, rf=rf, label=f"{bps} bps")
        results.append(stats)

    return pd.DataFrame(results)
