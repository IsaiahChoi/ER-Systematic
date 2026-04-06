"""
universe.py
===========
Dynamic universe construction for the semiconductor research framework.

At each rebalance date the eligible universe is determined by applying
liquidity and history filters.  This avoids including stocks that did
not yet have sufficient history or market cap at the time of the signal.

Note on survivorship bias
--------------------------
The starting universe (``config.TICKERS``) is determined as of 2024 and
therefore contains survivorship bias — companies that were delisted or
went bankrupt between 2014 and 2024 are excluded.  This upward-biases
historical backtest returns.  Production research would use a point-in-time
database (e.g. Compustat CRSP universe, FactSet entity history).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def apply_filters(
    prices: pd.DataFrame,
    market_caps: Optional[pd.DataFrame],
    min_mcap: float = 1e9,
    min_history: int = 252,
    min_price: float = 1.0,
) -> dict[pd.Timestamp, list[str]]:
    """Determine eligible tickers at each rebalance date.

    A stock is included in the universe at date ``t`` if, as of ``t``:
      1. It has at least ``min_history`` trading days of price history.
      2. Its most recent closing price is above ``min_price`` (removes
         penny stocks and near-delisted names).
      3. Its estimated market cap exceeds ``min_mcap`` (default: $1 billion).
         If ``market_caps`` is None, the market-cap filter is skipped.

    Args:
        prices: DataFrame of adjusted close prices (dates × tickers).
        market_caps: Optional DataFrame of estimated market caps
            (dates × tickers).  If None, market cap filter is skipped.
        min_mcap: Minimum market cap in USD (default: $1 billion).
        min_history: Minimum number of trading days of history required
            (default: 252, approximately one year).
        min_price: Minimum stock price in USD (default: $1.00).

    Returns:
        Dictionary mapping each rebalance date (``pd.Timestamp``) to the
        list of eligible ticker strings as of that date.
    """
    universe_by_date: dict[pd.Timestamp, list[str]] = {}
    all_dates = prices.index

    for date in all_dates:
        eligible: list[str] = []
        # Slice price history up to and including this date
        hist = prices.loc[:date]

        for ticker in prices.columns:
            col = hist[ticker].dropna()

            # Filter 1: Sufficient price history
            if len(col) < min_history:
                continue

            # Filter 2: Minimum price (not a penny stock)
            last_price = col.iloc[-1]
            if np.isnan(last_price) or last_price < min_price:
                continue

            # Filter 3: Market cap (if available)
            if market_caps is not None and ticker in market_caps.columns:
                mcap_slice = market_caps.loc[:date, ticker].dropna()
                if len(mcap_slice) > 0:
                    last_mcap = mcap_slice.iloc[-1]
                    if np.isnan(last_mcap) or last_mcap < min_mcap:
                        continue

            eligible.append(ticker)

        universe_by_date[date] = eligible

    return universe_by_date


def get_eligible_tickers(
    universe_by_date: dict[pd.Timestamp, list[str]],
    date: pd.Timestamp,
) -> list[str]:
    """Look up eligible tickers for a given date.

    If the exact date is not in the dictionary (e.g., non-rebalance day),
    this returns the most recent prior entry.

    Args:
        universe_by_date: Output of ``apply_filters``.
        date: Query date.

    Returns:
        List of eligible tickers, or empty list if no prior entry exists.
    """
    if date in universe_by_date:
        return universe_by_date[date]
    # Find most recent prior date
    prior_dates = [d for d in universe_by_date if d <= date]
    if not prior_dates:
        return []
    return universe_by_date[max(prior_dates)]


def build_rebalance_universe(
    prices: pd.DataFrame,
    market_caps: Optional[pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    min_mcap: float = 1e9,
    min_history: int = 252,
    min_price: float = 1.0,
) -> dict[pd.Timestamp, list[str]]:
    """Apply universe filters only at rebalance dates for efficiency.

    This is a faster version of ``apply_filters`` that only evaluates
    eligibility on the specific rebalance dates rather than every trading day.

    Args:
        prices: DataFrame of adjusted close prices.
        market_caps: Optional market cap DataFrame.
        rebalance_dates: DatetimeIndex of rebalance dates.
        min_mcap: Minimum market cap.
        min_history: Minimum price history (trading days).
        min_price: Minimum stock price.

    Returns:
        Dict mapping each rebalance date → list of eligible tickers.
    """
    universe_by_date: dict[pd.Timestamp, list[str]] = {}

    for date in rebalance_dates:
        # Align to the nearest available price date
        valid_dates = prices.index[prices.index <= date]
        if len(valid_dates) == 0:
            universe_by_date[date] = []
            continue
        actual_date = valid_dates[-1]

        eligible: list[str] = []
        hist = prices.loc[:actual_date]

        for ticker in prices.columns:
            col = hist[ticker].dropna()
            if len(col) < min_history:
                continue
            last_price = col.iloc[-1]
            if np.isnan(last_price) or last_price < min_price:
                continue

            if market_caps is not None and ticker in market_caps.columns:
                mcap_slice = market_caps.loc[:actual_date, ticker].dropna()
                if len(mcap_slice) > 0:
                    last_mcap = mcap_slice.iloc[-1]
                    if np.isnan(last_mcap) or last_mcap < min_mcap:
                        continue

            eligible.append(ticker)

        universe_by_date[date] = eligible

    return universe_by_date


def compute_universe_coverage(
    universe_by_date: dict[pd.Timestamp, list[str]],
    all_tickers: list[str],
) -> pd.DataFrame:
    """Compute summary statistics about universe coverage over time.

    Args:
        universe_by_date: Output of ``build_rebalance_universe``.
        all_tickers: Full candidate ticker list.

    Returns:
        DataFrame indexed by rebalance date with columns:
          - ``n_eligible``: Number of eligible tickers.
          - ``pct_eligible``: Fraction of the full candidate universe eligible.
          - ``eligible_tickers``: Comma-separated list of eligible tickers.
    """
    records = []
    for date, tickers in sorted(universe_by_date.items()):
        records.append({
            "date": date,
            "n_eligible": len(tickers),
            "pct_eligible": len(tickers) / max(len(all_tickers), 1),
            "eligible_tickers": ", ".join(sorted(tickers)),
        })
    return pd.DataFrame(records).set_index("date")
