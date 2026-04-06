"""
portfolio.py
============
Portfolio construction module.

Takes factor scores as input and produces portfolio weight dictionaries
that can be fed directly into the backtesting engine (analytics.py).

Three strategies are implemented:
  1. ``long_only_top_n``  — Equal-weight long-only top-N stocks.
  2. ``rank_and_select``  — Dollar-neutral long-short (top N vs. bottom N).
  3. A benchmark passthrough for SMH / SPY comparison.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils import cross_sectional_zscore, equal_weight, dollar_neutral_weights


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def composite_score(
    factor_df: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> pd.Series:
    """Combine individual factor z-scores into a single composite signal.

    Each factor is re-z-scored cross-sectionally (in case the input is not
    already on a uniform scale), then combined via a weighted average.

    Args:
        factor_df: DataFrame of factor scores (tickers × factors).
            Expected columns: momentum_6_1, earnings_revision,
            gross_margin_trend, rd_intensity_rank, relative_value,
            quality_composite.
        weights: Optional dict mapping factor name → relative weight.
            Weights are normalized to sum to 1.
            If None (default), all factors receive equal weight.

    Returns:
        pd.Series indexed by ticker with composite z-scores.
        Tickers with fewer than 2 valid factor scores are returned as NaN.

    Raises:
        ValueError: If ``weights`` specifies factor names not in ``factor_df``.
    """
    if factor_df.empty:
        return pd.Series(dtype=float)

    # Validate weight keys
    if weights is not None:
        unknown = set(weights) - set(factor_df.columns)
        if unknown:
            raise ValueError(
                f"Unknown factor names in weights: {unknown}. "
                f"Available: {list(factor_df.columns)}"
            )

    # Re-z-score each factor column cross-sectionally for uniformity
    zscored = factor_df.copy().astype(float)
    for col in zscored.columns:
        zscored[col] = cross_sectional_zscore(zscored[col])

    # Resolve weights
    if weights is None:
        # Equal weight all available columns
        w = {col: 1.0 / len(zscored.columns) for col in zscored.columns}
    else:
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Factor weights must sum to a positive number.")
        w = {col: weights[col] / total for col in weights if col in zscored.columns}

    # Weighted average, ignoring NaN (need at least 2 valid factors)
    scores = pd.Series(0.0, index=zscored.index)
    counts = pd.Series(0.0, index=zscored.index)

    for col, wt in w.items():
        valid = zscored[col].notna()
        scores[valid] += zscored.loc[valid, col] * wt
        counts[valid] += wt

    # Normalize by actual weight coverage (handle missing factors gracefully)
    scores = scores / counts.replace(0, np.nan)

    # Mask tickers with fewer than 2 valid factor observations
    n_valid = zscored.notna().sum(axis=1)
    scores[n_valid < 2] = np.nan

    scores.name = "composite_score"
    return scores


# ---------------------------------------------------------------------------
# Portfolio selection
# ---------------------------------------------------------------------------

def long_only_top_n(
    factor_scores: pd.Series,
    n: int = 10,
    weight_scheme: str = "equal",
) -> dict[str, float]:
    """Select the top-N stocks by composite score for a long-only portfolio.

    This is the primary strategy compared against the SMH benchmark.
    Equal weighting is used to avoid size bias and to keep rebalancing
    simple — in a real fund, weights might be tilted by conviction score
    or volatility-scaled.

    Args:
        factor_scores: Series of composite scores indexed by ticker.
            Higher score = more attractive.
        n: Number of stocks to select (default: 10).
        weight_scheme: ``"equal"`` (default) or ``"score"`` (score-proportional).

    Returns:
        Dict mapping selected ticker → portfolio weight (all positive,
        summing to ~1.0).

    Raises:
        ValueError: If ``weight_scheme`` is not recognized.
    """
    valid = factor_scores.dropna().sort_values(ascending=False)

    if len(valid) == 0:
        return {}

    n_actual = min(n, len(valid))
    selected = valid.head(n_actual)

    if weight_scheme == "equal":
        return equal_weight(list(selected.index))
    elif weight_scheme == "score":
        # Shift scores so they are all positive, then normalize
        shifted = selected - selected.min() + 1e-6
        total = shifted.sum()
        return (shifted / total).to_dict()
    else:
        raise ValueError(f"Unknown weight_scheme '{weight_scheme}'. Use 'equal' or 'score'.")


def rank_and_select(
    factor_scores: pd.Series,
    n_long: int = 10,
    n_short: int = 10,
    long_weight: float = 0.5,
    short_weight: float = 0.5,
) -> dict[str, float]:
    """Construct a dollar-neutral long-short portfolio.

    The long leg holds the top-N stocks; the short leg holds the bottom-N.
    By default, gross exposure is 1 (0.5 long + 0.5 short) and the
    portfolio is dollar-neutral.

    Args:
        factor_scores: Composite scores indexed by ticker.
        n_long: Number of long positions (default: 10).
        n_short: Number of short positions (default: 10).
        long_weight: Total long-side weight (default: 0.5).
        short_weight: Total short-side weight (default: 0.5).

    Returns:
        Dict of ticker → signed weight (+long, −short).
        Sum of absolute weights = ``long_weight + short_weight``.
    """
    valid = factor_scores.dropna().sort_values(ascending=False)

    if len(valid) < 2:
        return {}

    n_long_actual  = min(n_long,  len(valid) // 2)
    n_short_actual = min(n_short, len(valid) // 2)

    long_tickers  = list(valid.head(n_long_actual).index)
    short_tickers = list(valid.tail(n_short_actual).index)

    # Remove overlap (shouldn't happen but defensive)
    overlap = set(long_tickers) & set(short_tickers)
    long_tickers  = [t for t in long_tickers  if t not in overlap]
    short_tickers = [t for t in short_tickers if t not in overlap]

    weights: dict[str, float] = {}
    if long_tickers:
        lw = long_weight / len(long_tickers)
        for t in long_tickers:
            weights[t] = lw
    if short_tickers:
        sw = short_weight / len(short_tickers)
        for t in short_tickers:
            weights[t] = -sw

    return weights


def bottom_n_weights(
    factor_scores: pd.Series,
    n: int = 10,
) -> dict[str, float]:
    """Select bottom-N stocks (for short leg analysis), returned as positive weights.

    Args:
        factor_scores: Composite scores indexed by ticker.
        n: Number of bottom stocks.

    Returns:
        Dict of ticker → weight (positive, for analysis — not signed short).
    """
    valid = factor_scores.dropna().sort_values(ascending=True)
    n_actual = min(n, len(valid))
    selected = valid.head(n_actual)
    return equal_weight(list(selected.index))


# ---------------------------------------------------------------------------
# Portfolio construction through time
# ---------------------------------------------------------------------------

def build_weights_history(
    factor_panel: dict[pd.Timestamp, pd.DataFrame],
    strategy: str = "long_only",
    n_long: int = 10,
    n_short: int = 10,
    factor_weights: Optional[dict[str, float]] = None,
    weight_scheme: str = "equal",
) -> dict[pd.Timestamp, dict[str, float]]:
    """Build a time-series of portfolio weights at each rebalance date.

    Args:
        factor_panel: Dict from ``factors.build_factor_panel``.
            Maps rebalance date → factor DataFrame (tickers × factors).
        strategy: One of ``"long_only"`` or ``"long_short"``.
        n_long: Number of long positions.
        n_short: Number of short positions (only used for ``"long_short"``).
        factor_weights: Optional per-factor weights passed to
            ``composite_score``.  None = equal weight.
        weight_scheme: Weight scheme for long-only (``"equal"`` or
            ``"score"``).

    Returns:
        Dict mapping each rebalance date → portfolio weight dict.
    """
    weights_history: dict[pd.Timestamp, dict[str, float]] = {}

    for date, factor_df in sorted(factor_panel.items()):
        if factor_df is None or factor_df.empty:
            weights_history[date] = {}
            continue

        # Compute composite score
        scores = composite_score(factor_df, weights=factor_weights)

        if strategy == "long_only":
            w = long_only_top_n(scores, n=n_long, weight_scheme=weight_scheme)
        elif strategy == "long_short":
            w = rank_and_select(scores, n_long=n_long, n_short=n_short)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'long_only' or 'long_short'.")

        weights_history[date] = w

    return weights_history


def get_portfolio_holdings(
    weights_history: dict[pd.Timestamp, dict[str, float]],
    date: pd.Timestamp,
) -> dict[str, float]:
    """Get portfolio holdings on or before a given date.

    Args:
        weights_history: Time-series of portfolio weights.
        date: Query date.

    Returns:
        Weight dict for the most recent rebalance on or before ``date``.
    """
    eligible_dates = [d for d in weights_history if d <= date]
    if not eligible_dates:
        return {}
    return weights_history[max(eligible_dates)]


def summarize_holdings_table(
    factor_panel: dict[pd.Timestamp, pd.DataFrame],
    weights_history: dict[pd.Timestamp, dict[str, float]],
    ticker_names: dict[str, str],
    ticker_segments: dict[str, str],
    date: Optional[pd.Timestamp] = None,
    n_show: int = 10,
) -> pd.DataFrame:
    """Produce a formatted holdings table for the most recent rebalance.

    Shows ticker, company name, sub-segment, factor scores, composite
    score, and portfolio weight for the top-N and bottom-N stocks.

    Args:
        factor_panel: Dict of factor DataFrames.
        weights_history: Dict of portfolio weight dicts.
        ticker_names: Mapping of ticker → full company name.
        ticker_segments: Mapping of ticker → sub-segment.
        date: Target date.  If None, uses the most recent rebalance.
        n_show: Number of top/bottom holdings to show.

    Returns:
        DataFrame suitable for display in the research notebook.
    """
    if date is None:
        date = max(factor_panel.keys())

    factor_df = factor_panel.get(date, pd.DataFrame())
    weights = weights_history.get(date, {})

    if factor_df.empty:
        return pd.DataFrame()

    scores = composite_score(factor_df)
    scores_df = scores.rename("composite_score").to_frame()

    # Merge factor scores
    result = factor_df.join(scores_df, how="outer")
    result["weight"] = result.index.map(lambda t: weights.get(t, 0.0))
    result["company"] = result.index.map(lambda t: ticker_names.get(t, t))
    result["segment"] = result.index.map(lambda t: ticker_segments.get(t, "Unknown"))

    result = result.sort_values("composite_score", ascending=False)

    cols = ["company", "segment", "momentum_6_1", "earnings_revision",
            "gross_margin_trend", "rd_intensity_rank", "relative_value",
            "quality_composite", "composite_score", "weight"]
    available_cols = [c for c in cols if c in result.columns]

    return result[available_cols]
