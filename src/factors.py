"""
factors.py
==========
Alpha factor construction for the semiconductor equity research framework.

Six factors are implemented, each grounded in a specific economic hypothesis
about what drives returns in the semiconductor sector:

  1. ``momentum_6_1``       — Price momentum (6-month, skip-1-month)
  2. ``earnings_revision``  — EPS trajectory proxy (IBES substitute)
  3. ``gross_margin_trend`` — Margin expansion / contraction signal
  4. ``rd_intensity_rank``  — Innovation pipeline quality
  5. ``relative_value``     — Sector-relative EV/Sales valuation
  6. ``quality_composite``  — Multi-metric quality score

Look-ahead bias prevention
---------------------------
All fundamental-based factors retrieve the most recent available quarter
whose ``period_end`` date falls at least ``filing_lag_days`` before the
signal date.  This approximates the 45-day lag between a fiscal quarter
end and the 10-Q/10-K filing date.

Design conventions
-------------------
- Every factor function returns a ``pd.Series`` indexed by ticker symbol,
  with higher values meaning a *more attractive* signal (long-favoring).
- NaN is returned for tickers with insufficient data.
- All series are z-scored cross-sectionally inside ``compute_all_factors``;
  individual factor functions return raw values.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils import cross_sectional_zscore, percentile_rank


# ---------------------------------------------------------------------------
# Individual factor functions
# ---------------------------------------------------------------------------

def momentum_6_1(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    skip_months: int = 1,
    lookback_months: int = 6,
) -> pd.Series:
    """Compute 6-month price momentum skipping the last 1 month.

    Economic rationale: Stock price momentum is one of the most robust
    cross-sectional return predictors documented in the academic literature
    (Jegadeesh & Titman, 1993).  In semiconductors, momentum captures
    the persistent revision cycle — when a chip company beats estimates,
    analysts typically revise numbers upward over multiple quarters, dragging
    the stock price higher with a lag.

    Construction:
        Return from (date − 7 months) to (date − 1 month), i.e., months
        t−7 to t−1.  The last month is skipped to avoid microstructure
        mean-reversion (bid-ask bounce).

    Args:
        prices: DataFrame of adjusted close prices (dates × tickers).
        date: Signal date (the rebalance date).
        skip_months: Number of months to skip at the end (default: 1).
        lookback_months: Lookback window in months, excluding the skip
            period (default: 6).

    Returns:
        Series indexed by ticker with 6-1 momentum returns.  NaN for
        tickers with insufficient history.
    """
    total_months = lookback_months + skip_months

    # End of the momentum window = date minus skip period
    end_date = date - pd.DateOffset(months=skip_months)
    start_date = date - pd.DateOffset(months=total_months)

    # Snap to actual available dates in the price series
    available = prices.index
    end_idx = available[available <= end_date]
    start_idx = available[available <= start_date]

    if len(end_idx) == 0 or len(start_idx) == 0:
        return pd.Series(np.nan, index=prices.columns)

    end_actual = end_idx[-1]
    start_actual = start_idx[-1]

    if start_actual >= end_actual:
        return pd.Series(np.nan, index=prices.columns)

    p_end = prices.loc[end_actual]
    p_start = prices.loc[start_actual]

    returns = (p_end / p_start) - 1.0
    returns.name = "momentum_6_1"
    return returns


def earnings_revision(
    fundamentals: pd.DataFrame,
    date: pd.Timestamp,
    filing_lag_days: int = 45,
    n_quarters_back: int = 4,
) -> pd.Series:
    """Proxy for analyst EPS estimate revision using trailing EPS growth.

    **NOTE**: A production implementation would use IBES/FactSet consensus
    EPS estimate revisions (% change in 12-month forward EPS estimate over
    the past 1–3 months).  That data is not freely available.  We proxy it
    as the year-over-year change in diluted EPS, which captures the same
    underlying fundamental improvement but with more noise and a longer lag.

    Economic rationale: Stocks with improving fundamental earnings momentum
    — reflected in rising analyst estimates — consistently outperform over
    3–6 month horizons.  In the semiconductor sector, the earnings revision
    cycle is amplified by the industry's extreme operating leverage: revenue
    beats at cyclical peaks translate into outsized EPS upside due to high
    fixed costs.

    Construction:
        EPS_revision = (EPS_latest_quarter − EPS_same_quarter_1_year_ago)
                       / |EPS_same_quarter_1_year_ago|

        A positive value means EPS improved year-over-year.

    Args:
        fundamentals: MultiIndex DataFrame (ticker, period_end) from
            ``data_loader.compute_derived_fundamentals``.
        date: Signal date.
        filing_lag_days: Days to lag from period_end (avoids look-ahead).
        n_quarters_back: Number of quarters for YoY comparison (default: 4).

    Returns:
        Series indexed by ticker.  Higher = better (EPS improving).
    """
    cutoff = date - pd.Timedelta(days=filing_lag_days)
    result: dict[str, float] = {}

    if fundamentals.empty:
        return pd.Series(dtype=float)

    for ticker, grp in fundamentals.groupby(level="ticker"):
        # Only use data available as of the signal date
        available = grp.loc[:, "eps_diluted"].reset_index(level="ticker", drop=True)
        available = available[available.index <= cutoff].dropna()

        if len(available) < n_quarters_back + 1:
            result[ticker] = np.nan
            continue

        eps_latest = available.iloc[-1]
        eps_year_ago = available.iloc[-(n_quarters_back + 1)]

        denom = abs(eps_year_ago)
        if denom < 1e-6:
            result[ticker] = np.nan
            continue

        result[ticker] = (eps_latest - eps_year_ago) / denom

    series = pd.Series(result, name="earnings_revision")
    return series


def gross_margin_trend(
    fundamentals: pd.DataFrame,
    date: pd.Timestamp,
    filing_lag_days: int = 45,
    n_quarters: int = 4,
) -> pd.Series:
    """Trailing 4-quarter change in gross margin (in basis points).

    Economic rationale: Gross margin expansion in semiconductors signals
    a favorable product mix shift toward higher-ASP products (e.g., data
    center vs. consumer), improving pricing power, or better fab yields.
    Margin expansion tends to precede earnings beats and positive guidance
    revisions, making it a leading indicator of future outperformance.
    We measure the change in percentage points (× 100 = bps) to capture
    direction and magnitude.

    Construction:
        ΔGM = (GM_most_recent_quarter − GM_4_quarters_ago) × 10000 [bps]

    Args:
        fundamentals: MultiIndex DataFrame (ticker, period_end).
        date: Signal date.
        filing_lag_days: Filing lag in days.
        n_quarters: Number of quarters over which to measure change
            (default: 4 = one year).

    Returns:
        Series indexed by ticker.  Higher (more positive bps) = improving
        margins = more attractive.
    """
    cutoff = date - pd.Timedelta(days=filing_lag_days)
    result: dict[str, float] = {}

    for ticker, grp in fundamentals.groupby(level="ticker"):
        gm = grp["gross_margin"].reset_index(level="ticker", drop=True)
        gm = gm[gm.index <= cutoff].dropna()

        if len(gm) < n_quarters + 1:
            result[ticker] = np.nan
            continue

        gm_recent = gm.iloc[-1]
        gm_prior = gm.iloc[-(n_quarters + 1)]
        result[ticker] = (gm_recent - gm_prior) * 10_000  # convert to bps

    series = pd.Series(result, name="gross_margin_trend")
    return series


def rd_intensity_rank(
    fundamentals: pd.DataFrame,
    date: pd.Timestamp,
    filing_lag_days: int = 45,
    n_quarters: int = 2,
) -> pd.Series:
    """Cross-sectional percentile rank of R&D intensity (R&D / Revenue).

    Economic rationale: Semiconductors are an intensely R&D-driven industry
    where innovation cadence (product cycles, process node transitions) is
    the primary source of competitive moats.  High R&D intensity signals
    investment in future product generations, which historically leads to
    market share gains and margin expansion 2–4 years forward.  Within the
    sector, the most R&D-intensive companies — relative to peers — tend to
    sustain premium valuations through technology leadership.

    **Note**: R&D intensity is a quality/innovation signal, not a value
    signal.  High-intensity firms spend more today but can compound faster.
    The relationship is non-linear: extremely low R&D (commodity play) and
    extremely high R&D (pre-revenue) both carry risk.

    Construction:
        Raw = TTM R&D Expense / TTM Revenue (trailing 2-quarter average).
        Signal = cross-sectional percentile rank (0 = lowest, 1 = highest).

    Args:
        fundamentals: MultiIndex DataFrame (ticker, period_end).
        date: Signal date.
        filing_lag_days: Filing lag in days.
        n_quarters: Number of recent quarters to average for TTM proxy.

    Returns:
        Series indexed by ticker with percentile rank (0–1).
    """
    cutoff = date - pd.Timedelta(days=filing_lag_days)
    raw_values: dict[str, float] = {}

    for ticker, grp in fundamentals.groupby(level="ticker"):
        rd = grp["rd_intensity"].reset_index(level="ticker", drop=True)
        rd = rd[rd.index <= cutoff].dropna()

        if len(rd) < 1:
            raw_values[ticker] = np.nan
            continue

        # Use average of most recent n_quarters for stability
        raw_values[ticker] = rd.iloc[-n_quarters:].mean()

    raw_series = pd.Series(raw_values, name="rd_intensity_raw")
    # Cross-sectional percentile rank (higher R&D intensity = higher rank)
    ranked = percentile_rank(raw_series)
    ranked.name = "rd_intensity_rank"
    return ranked


def relative_value(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    date: pd.Timestamp,
    filing_lag_days: int = 45,
    n_quarters: int = 4,
) -> pd.Series:
    """Sector-relative EV/Sales z-score (inverted so cheaper = higher signal).

    Economic rationale: Valuation discipline matters even in high-growth
    sectors.  In semiconductors, the EV/Sales multiple is preferred over
    P/E because: (1) margins are highly volatile through the cycle, making
    earnings-based multiples noisy; and (2) revenue is a cleaner signal of
    business scale.  We compute the sector-relative z-score — how cheap is
    each stock relative to its peers — and invert it so cheaper stocks score
    higher.  This is a contrarian/mean-reversion signal best used in
    combination with quality and momentum factors.

    Construction:
        Revenue_TTM = sum of last 4 quarters of revenue.
        Price = closing price on the signal date.
        Market Cap ≈ Price × Shares Outstanding.
        EV ≈ Market Cap + Total Debt − Cash.
        EV_Sales = EV / Revenue_TTM.
        Signal = −z-score(EV_Sales) [inverted: cheaper = positive].

    Args:
        fundamentals: MultiIndex DataFrame (ticker, period_end).
        prices: Adjusted close price DataFrame (dates × tickers).
        date: Signal date.
        filing_lag_days: Filing lag in days.
        n_quarters: Number of quarters to sum for trailing revenue.

    Returns:
        Series indexed by ticker.  Higher = cheaper relative to peers.
    """
    cutoff = date - pd.Timedelta(days=filing_lag_days)
    ev_sales: dict[str, float] = {}

    # Get most recent available price
    price_date = prices.index[prices.index <= date]
    if len(price_date) == 0:
        return pd.Series(np.nan, dtype=float)
    price_date = price_date[-1]
    current_prices = prices.loc[price_date]

    for ticker, grp in fundamentals.groupby(level="ticker"):
        available = grp[grp.index.get_level_values("period_end") <= cutoff]
        if len(available) < n_quarters:
            ev_sales[ticker] = np.nan
            continue

        recent = available.iloc[-n_quarters:]
        ttm_revenue = recent["revenue"].sum()
        if ttm_revenue <= 0 or np.isnan(ttm_revenue):
            ev_sales[ticker] = np.nan
            continue

        # Latest fundamental snapshot
        latest = available.iloc[-1]
        shares = latest.get("shares_outstanding") if hasattr(latest, "get") else np.nan
        try:
            shares = float(latest["shares_outstanding"])
        except (KeyError, TypeError, ValueError):
            shares = np.nan

        if np.isnan(shares) or shares <= 0:
            ev_sales[ticker] = np.nan
            continue

        if ticker not in current_prices.index or np.isnan(current_prices[ticker]):
            ev_sales[ticker] = np.nan
            continue

        price = float(current_prices[ticker])
        market_cap = price * shares

        try:
            total_debt = float(latest["total_debt"]) if not np.isnan(latest["total_debt"]) else 0.0
        except (KeyError, TypeError, ValueError):
            total_debt = 0.0

        try:
            cash = float(latest["cash_and_equivalents"]) if not np.isnan(latest["cash_and_equivalents"]) else 0.0
        except (KeyError, TypeError, ValueError):
            cash = 0.0

        ev = market_cap + total_debt - cash
        ev_sales[ticker] = ev / ttm_revenue

    raw = pd.Series(ev_sales, name="ev_sales_raw")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    # Invert z-score: cheaper (lower EV/Sales) → higher signal
    z = cross_sectional_zscore(raw)
    inverted = -z
    inverted.name = "relative_value"
    return inverted


def quality_composite(
    fundamentals: pd.DataFrame,
    date: pd.Timestamp,
    filing_lag_days: int = 45,
    n_quarters: int = 4,
) -> pd.Series:
    """Multi-metric quality composite score.

    Economic rationale: Quality investing — buying companies with strong
    fundamental characteristics — has documented historical outperformance
    (Novy-Marx, 2013; Asness, Frazzini & Pedersen, 2019).  In semiconductors,
    quality is particularly important because the industry is highly cyclical:
    high-quality firms (strong margins, efficient capital use, growing revenue)
    are better positioned to invest through downturns and emerge with increased
    market share.  The quality composite combines three complementary metrics
    that together capture profitability, capital efficiency, and growth.

    Construction:
        Component 1 — Gross Margin: Most recent quarter gross margin.
        Component 2 — ROIC: Most recent quarter return on invested capital.
        Component 3 — Revenue Growth YoY: Year-over-year quarterly revenue
            growth rate.

        Each component is z-scored cross-sectionally.  The quality composite
        is the equal-weight average of the three z-scores.

    Args:
        fundamentals: MultiIndex DataFrame (ticker, period_end).
        date: Signal date.
        filing_lag_days: Filing lag in days.
        n_quarters: Number of recent quarters to average each metric.

    Returns:
        Series indexed by ticker.  Higher = better quality.
    """
    cutoff = date - pd.Timedelta(days=filing_lag_days)

    metrics: dict[str, dict[str, float]] = {
        "gross_margin": {},
        "roic": {},
        "revenue_growth_yoy": {},
    }

    for ticker, grp in fundamentals.groupby(level="ticker"):
        available = grp[grp.index.get_level_values("period_end") <= cutoff]
        if len(available) == 0:
            for k in metrics:
                metrics[k][ticker] = np.nan
            continue

        for metric in metrics:
            col = available[metric].dropna()
            if len(col) == 0:
                metrics[metric][ticker] = np.nan
            else:
                # Use most recent observation (or mean of last n_quarters for smoothing)
                metrics[metric][ticker] = col.iloc[-min(n_quarters, len(col)):].mean()

    # Z-score each component cross-sectionally
    z_scores = {}
    for metric, vals in metrics.items():
        s = pd.Series(vals)
        z_scores[metric] = cross_sectional_zscore(s)

    # Equal-weight composite
    composite = pd.DataFrame(z_scores).mean(axis=1)
    composite.name = "quality_composite"
    return composite


# ---------------------------------------------------------------------------
# Composite factor aggregator
# ---------------------------------------------------------------------------

def compute_all_factors(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    date: pd.Timestamp,
    eligible_tickers: Optional[list[str]] = None,
    filing_lag_days: int = 45,
) -> pd.DataFrame:
    """Compute all 6 factor scores for the eligible universe at a given date.

    Each factor is z-scored cross-sectionally before being returned.
    This ensures all factors are on the same scale for composite construction.

    Args:
        prices: Adjusted close price DataFrame (dates × tickers).
        fundamentals: MultiIndex fundamentals DataFrame (ticker, period_end).
        date: Rebalance / signal date.
        eligible_tickers: Optional list of tickers to restrict the universe.
            If None, all price columns are used.
        filing_lag_days: Filing lag in days for fundamental data.

    Returns:
        DataFrame indexed by ticker with columns:
          momentum_6_1, earnings_revision, gross_margin_trend,
          rd_intensity_rank, relative_value, quality_composite.
        All values are cross-sectional z-scores.  NaN for insufficient data.
    """
    if eligible_tickers is not None:
        price_df = prices[eligible_tickers] if eligible_tickers else prices.copy()
    else:
        price_df = prices.copy()

    # Restrict fundamentals to eligible tickers if provided
    if eligible_tickers and not fundamentals.empty:
        eligible_mask = fundamentals.index.get_level_values("ticker").isin(eligible_tickers)
        fund_df = fundamentals.loc[eligible_mask]
    else:
        fund_df = fundamentals.copy()

    factors: dict[str, pd.Series] = {}

    # 1. Momentum
    mom = momentum_6_1(price_df, date)
    factors["momentum_6_1"] = cross_sectional_zscore(mom)

    # 2. Earnings revision
    if not fund_df.empty:
        er = earnings_revision(fund_df, date, filing_lag_days)
        factors["earnings_revision"] = cross_sectional_zscore(er)
    else:
        factors["earnings_revision"] = pd.Series(np.nan, index=price_df.columns)

    # 3. Gross margin trend
    if not fund_df.empty:
        gmt = gross_margin_trend(fund_df, date, filing_lag_days)
        factors["gross_margin_trend"] = cross_sectional_zscore(gmt)
    else:
        factors["gross_margin_trend"] = pd.Series(np.nan, index=price_df.columns)

    # 4. R&D intensity rank
    if not fund_df.empty:
        rdi = rd_intensity_rank(fund_df, date, filing_lag_days)
        factors["rd_intensity_rank"] = rdi  # already ranked; z-score for consistency
        factors["rd_intensity_rank"] = cross_sectional_zscore(rdi)
    else:
        factors["rd_intensity_rank"] = pd.Series(np.nan, index=price_df.columns)

    # 5. Relative value
    if not fund_df.empty:
        rv = relative_value(fund_df, price_df, date, filing_lag_days)
        factors["relative_value"] = rv  # already z-scored inside function
    else:
        factors["relative_value"] = pd.Series(np.nan, index=price_df.columns)

    # 6. Quality composite
    if not fund_df.empty:
        qc = quality_composite(fund_df, date, filing_lag_days)
        factors["quality_composite"] = qc  # already z-scored inside function
    else:
        factors["quality_composite"] = pd.Series(np.nan, index=price_df.columns)

    factor_df = pd.DataFrame(factors)
    factor_df.index.name = "ticker"

    # Restrict to eligible tickers
    if eligible_tickers:
        present = [t for t in eligible_tickers if t in factor_df.index]
        factor_df = factor_df.loc[present]

    return factor_df


def build_factor_panel(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    eligible_universe: dict[pd.Timestamp, list[str]],
    filing_lag_days: int = 45,
    verbose: bool = True,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Build the full factor panel across all rebalance dates.

    Args:
        prices: Adjusted close price DataFrame.
        fundamentals: Derived fundamentals DataFrame (MultiIndex).
        rebalance_dates: Rebalance date index.
        eligible_universe: Dict from ``universe.build_rebalance_universe``.
        filing_lag_days: Filing lag.
        verbose: Print progress.

    Returns:
        Dict mapping each rebalance date → factor DataFrame
        (tickers × factors, z-scored cross-sectionally).
    """
    panel: dict[pd.Timestamp, pd.DataFrame] = {}

    for i, date in enumerate(rebalance_dates):
        if verbose and (i % 12 == 0 or i == len(rebalance_dates) - 1):
            print(f"  Computing factors: {date.date()} ({i+1}/{len(rebalance_dates)})")

        eligible = eligible_universe.get(date, list(prices.columns))

        try:
            factor_df = compute_all_factors(
                prices=prices,
                fundamentals=fundamentals,
                date=date,
                eligible_tickers=eligible,
                filing_lag_days=filing_lag_days,
            )
            panel[date] = factor_df
        except Exception as exc:
            if verbose:
                print(f"  Warning: factor computation failed at {date}: {exc}")
            panel[date] = pd.DataFrame()

    return panel
