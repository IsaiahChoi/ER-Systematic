"""
data_loader.py
==============
Handles all external data acquisition:
  - Equity prices (adjusted close) via yfinance
  - Quarterly fundamental data via yfinance
  - Macro series via fredapi (FRED)
  - Derived fundamental metrics (margins, growth, ROIC, etc.)

No look-ahead bias is introduced here; callers are responsible for lagging
fundamental data (see ``config.FILING_LAG_DAYS`` and ``utils.lag_fundamentals``).
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    progress: bool = False,
    retries: int = 3,
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers.

    Uses yfinance's ``download`` function.  Handles multi-ticker vs.
    single-ticker return format differences transparently.

    Args:
        tickers: List of ticker symbols (e.g. ``["NVDA", "AMD", "SMH"]``).
        start: Start date string ``"YYYY-MM-DD"``.
        end: End date string ``"YYYY-MM-DD"``.
        progress: Whether to show yfinance download progress bar.
        retries: Number of download retry attempts on failure.

    Returns:
        DataFrame of adjusted close prices, indexed by date, one column per
        ticker.  Missing tickers (delisted, unavailable) will have NaN columns.
    """
    all_tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers=all_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=progress,
                threads=True,
            )
            break
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to download prices after {retries} attempts: {exc}") from exc

    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        # Level 0 = price type (Close, Open, …), Level 1 = ticker
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            prices = raw.xs("Close", axis=1, level=0)
    else:
        # Single ticker — column name is the price type
        prices = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    # Ensure all requested tickers are present (fill missing with NaN)
    for t in all_tickers:
        if t not in prices.columns:
            prices[t] = np.nan

    return prices[all_tickers].copy()


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

def fetch_fundamentals(
    tickers: list[str],
    progress: bool = True,
) -> pd.DataFrame:
    """Fetch quarterly financial statement data for each ticker via yfinance.

    Pulls income statement, balance sheet, and cash flow data.
    Returns a "long" DataFrame indexed by (ticker, period_end_date) with
    columns for key fundamental metrics.

    Items collected per quarter:
        - revenue (Total Revenue)
        - gross_profit
        - operating_income (Operating Income / EBIT)
        - net_income
        - rd_expense (Research and Development)
        - capex (Capital Expenditure, signed positive = outflow)
        - eps_diluted (Diluted EPS)
        - shares_outstanding (Diluted shares)
        - total_assets
        - total_debt (Short + Long term debt)
        - stockholders_equity
        - cash_and_equivalents

    Args:
        tickers: List of ticker symbols.
        progress: Print progress to stdout.

    Returns:
        DataFrame with MultiIndex (ticker, period_end_date), sorted.
        All monetary values in USD (as reported by yfinance).
    """
    records: list[dict] = []

    for i, ticker in enumerate(tickers):
        if progress:
            print(f"  [{i+1}/{len(tickers)}] Fetching fundamentals: {ticker}", end="\r")
        try:
            tk = yf.Ticker(ticker)
            # ---- Income Statement ----
            inc = tk.quarterly_income_stmt
            if inc is None or inc.empty:
                inc = tk.quarterly_financials  # fallback

            # ---- Balance Sheet ----
            bal = tk.quarterly_balance_sheet

            # ---- Cash Flow ----
            cf = tk.quarterly_cash_flow

            if inc is None or inc.empty:
                continue

            # Dates are columns in yfinance; rows are line items
            for col_date in inc.columns:
                row: dict = {"ticker": ticker, "period_end": pd.to_datetime(col_date)}

                # --- Income statement items ---
                row["revenue"] = _get_item(inc, col_date, [
                    "Total Revenue", "TotalRevenue", "Revenue",
                ])
                row["gross_profit"] = _get_item(inc, col_date, [
                    "Gross Profit", "GrossProfit",
                ])
                row["operating_income"] = _get_item(inc, col_date, [
                    "Operating Income", "OperatingIncome",
                    "EBIT", "Total Operating Income As Reported",
                ])
                row["net_income"] = _get_item(inc, col_date, [
                    "Net Income", "NetIncome",
                    "Net Income Common Stockholders",
                ])
                row["rd_expense"] = _get_item(inc, col_date, [
                    "Research And Development", "ResearchAndDevelopment",
                    "Research Development", "R&D Expenses",
                ])
                row["eps_diluted"] = _get_item(inc, col_date, [
                    "Diluted EPS", "DilutedEPS",
                    "Basic EPS", "EPS",
                ])

                # --- Balance sheet items ---
                if bal is not None and not bal.empty and col_date in bal.columns:
                    row["total_assets"] = _get_item(bal, col_date, [
                        "Total Assets", "TotalAssets",
                    ])
                    row["total_debt"] = _get_item(bal, col_date, [
                        "Total Debt", "TotalDebt", "Long Term Debt",
                        "Short Long Term Debt", "Current Debt",
                    ])
                    row["stockholders_equity"] = _get_item(bal, col_date, [
                        "Stockholders Equity", "StockholdersEquity",
                        "Total Equity Gross Minority Interest",
                        "Common Stock Equity",
                    ])
                    row["cash_and_equivalents"] = _get_item(bal, col_date, [
                        "Cash And Cash Equivalents", "CashAndCashEquivalents",
                        "Cash Cash Equivalents And Short Term Investments",
                    ])
                else:
                    row.update({
                        "total_assets": np.nan, "total_debt": np.nan,
                        "stockholders_equity": np.nan, "cash_and_equivalents": np.nan,
                    })

                # --- Cash flow items ---
                if cf is not None and not cf.empty and col_date in cf.columns:
                    capex_raw = _get_item(cf, col_date, [
                        "Capital Expenditure", "CapitalExpenditure",
                        "Purchase Of Property Plant And Equipment",
                        "Capital Expenditures",
                    ])
                    # yfinance returns capex as negative → make positive
                    row["capex"] = abs(capex_raw) if not np.isnan(capex_raw) else np.nan
                else:
                    row["capex"] = np.nan

                # Shares outstanding (from info dict — more reliable)
                row["shares_outstanding"] = np.nan

                records.append(row)

        except Exception as exc:
            if progress:
                print(f"\n  Warning: could not fetch {ticker}: {exc}")
            continue

    if progress:
        print()  # newline after progress

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["period_end"] = pd.to_datetime(df["period_end"])

    # Fetch shares outstanding separately from .info (more reliable)
    df = _enrich_shares_outstanding(df, tickers, progress=progress)

    df = df.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    df = df.set_index(["ticker", "period_end"])
    return df


def _get_item(
    df: pd.DataFrame,
    col: object,
    keys: list[str],
) -> float:
    """Try multiple row-name variants in a yfinance statement DataFrame.

    Args:
        df: Statement DataFrame (rows = items, columns = dates).
        col: The date column to extract.
        keys: Ordered list of possible row names to try.

    Returns:
        Scalar value (float) or ``np.nan`` if none found.
    """
    for key in keys:
        if key in df.index:
            val = df.loc[key, col]
            if not (isinstance(val, float) and np.isnan(val)):
                return float(val)
    return np.nan


def _enrich_shares_outstanding(
    df: pd.DataFrame,
    tickers: list[str],
    progress: bool = False,
) -> pd.DataFrame:
    """Backfill shares_outstanding from yfinance .info for each ticker.

    Args:
        df: Fundamentals DataFrame (un-indexed, with 'ticker' column).
        tickers: Ticker list.
        progress: Print progress.

    Returns:
        DataFrame with shares_outstanding filled where available.
    """
    shares_map: dict[str, float] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            if shares:
                shares_map[ticker] = float(shares)
        except Exception:
            pass

    def fill_shares(row: pd.Series) -> float:
        if np.isnan(row["shares_outstanding"]):
            return shares_map.get(row["ticker"], np.nan)
        return row["shares_outstanding"]

    df["shares_outstanding"] = df.apply(fill_shares, axis=1)
    return df


# ---------------------------------------------------------------------------
# Derived fundamentals
# ---------------------------------------------------------------------------

def compute_derived_fundamentals(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics from raw fundamentals.

    Derived metrics added as new columns:
        - ``gross_margin``: Gross Profit / Revenue
        - ``operating_margin``: Operating Income / Revenue
        - ``net_margin``: Net Income / Revenue
        - ``rd_intensity``: R&D Expense / Revenue
        - ``capex_intensity``: Capex / Revenue
        - ``revenue_growth_yoy``: YoY quarterly revenue growth
        - ``eps_growth_yoy``: YoY quarterly EPS growth
        - ``roic``: (Operating Income × (1 − tax_rate)) / (Total Debt + Equity − Cash)
          Simplified: operating_income / (total_debt + stockholders_equity)
        - ``net_debt``: Total Debt − Cash and Equivalents

    Args:
        fundamentals_df: DataFrame from ``fetch_fundamentals`` with
            MultiIndex (ticker, period_end).

    Returns:
        Same DataFrame with additional derived columns.  Original index
        (ticker, period_end) preserved.
    """
    df = fundamentals_df.copy().reset_index()

    # --- Margins ---
    df["gross_margin"] = df["gross_profit"] / df["revenue"].replace(0, np.nan)
    df["operating_margin"] = df["operating_income"] / df["revenue"].replace(0, np.nan)
    df["net_margin"] = df["net_income"] / df["revenue"].replace(0, np.nan)
    df["rd_intensity"] = df["rd_expense"] / df["revenue"].replace(0, np.nan)
    df["capex_intensity"] = df["capex"] / df["revenue"].replace(0, np.nan)

    # --- Net debt ---
    df["net_debt"] = df["total_debt"].fillna(0) - df["cash_and_equivalents"].fillna(0)

    # --- YoY growth: within each ticker, shift 4 quarters (YoY) ---
    df = df.sort_values(["ticker", "period_end"])

    def yoy_growth(group: pd.DataFrame, col: str) -> pd.Series:
        """Compute YoY percentage change within a ticker group."""
        shifted = group[col].shift(4)
        return (group[col] - shifted) / shifted.abs().replace(0, np.nan)

    growth_rev = []
    growth_eps = []
    for ticker, grp in df.groupby("ticker"):
        growth_rev.append(yoy_growth(grp, "revenue"))
        growth_eps.append(yoy_growth(grp, "eps_diluted"))

    df["revenue_growth_yoy"] = pd.concat(growth_rev)
    df["eps_growth_yoy"] = pd.concat(growth_eps)

    # --- ROIC (simplified) ---
    # ROIC = Operating Income / Invested Capital
    # Invested Capital ≈ Total Debt + Stockholders' Equity (book value)
    invested_capital = df["total_debt"].fillna(0) + df["stockholders_equity"].fillna(0)
    df["roic"] = df["operating_income"] / invested_capital.replace(0, np.nan)

    # Clip extreme outliers
    for col in ["gross_margin", "operating_margin", "net_margin",
                "rd_intensity", "capex_intensity", "roic",
                "revenue_growth_yoy", "eps_growth_yoy"]:
        df[col] = df[col].clip(lower=-5.0, upper=5.0)

    return df.set_index(["ticker", "period_end"])


# ---------------------------------------------------------------------------
# Macro data
# ---------------------------------------------------------------------------

def fetch_macro(
    fred_series_dict: dict[str, str],
    start: str,
    end: str,
    api_key: str = "",
) -> pd.DataFrame:
    """Fetch macro series from FRED and return as a daily DataFrame.

    Series are resampled to business-day frequency (forward-filled).
    When ``api_key`` is empty, the function first tries the environment
    variable ``FRED_API_KEY``.

    Args:
        fred_series_dict: Dict mapping friendly name → FRED series ID.
            Example: ``{"yield_curve": "T10Y2Y", "oil": "DCOILWTICO"}``.
        start: Start date string ``"YYYY-MM-DD"``.
        end: End date string ``"YYYY-MM-DD"``.
        api_key: FRED API key string.  Falls back to env var.

    Returns:
        DataFrame indexed by business-day dates, one column per series name.

    Raises:
        ImportError: If ``fredapi`` is not installed.
        ValueError: If no valid API key is found.
    """
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError(
            "Please install fredapi: pip install fredapi"
        ) from exc

    # Resolve API key
    resolved_key = api_key or os.environ.get("FRED_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "No FRED API key found.  Set config.FRED_API_KEY or the "
            "FRED_API_KEY environment variable.  Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    fred = Fred(api_key=resolved_key)
    date_range = pd.date_range(start=start, end=end, freq="B")
    result = pd.DataFrame(index=date_range)
    result.index.name = "date"

    for name, series_id in fred_series_dict.items():
        try:
            series = fred.get_series(series_id, observation_start=start, observation_end=end)
            series = series.reindex(date_range, method="ffill")
            result[name] = series
        except Exception as exc:
            print(f"  Warning: could not fetch FRED series '{series_id}' ({name}): {exc}")
            result[name] = np.nan

    return result


def simulate_semi_revenue_growth(
    macro_df: pd.DataFrame,
    start: str,
    end: str,
    noise_scale: float = 0.015,
    random_seed: int = 42,
) -> pd.Series:
    """Simulate a semiconductor industry revenue growth proxy series.

    **NOTE — SIMULATED DATA**: The Semiconductor Industry Association (SIA)
    publishes monthly global semiconductor sales, but this data is not freely
    available on FRED.  We simulate a *stylized* proxy by:
      1. Using FRED's Industrial Production: Business Equipment (IPBUSEQ) as
         the backbone signal — this series historically correlates ~0.7 with
         actual SIA sales growth.
      2. Adding mean-reverting Gaussian noise (σ = 1.5% monthly) to mimic
         the higher cyclicality of semiconductors vs. broad industrial output.

    This series should be treated as **illustrative only** and is clearly
    labeled as simulated in all notebook outputs.  For production research,
    replace with SIA monthly data or WSTS statistics.

    Args:
        macro_df: DataFrame from ``fetch_macro`` containing ``"biz_equipment"``
            column (FRED series IPBUSEQ).
        start: Start date of the desired series.
        end: End date of the desired series.
        noise_scale: Standard deviation of additive noise (monthly level).
        random_seed: NumPy random seed for reproducibility.

    Returns:
        Monthly pandas Series of simulated YoY semiconductor revenue growth,
        indexed by month-end dates.
    """
    rng = np.random.default_rng(random_seed)

    if "biz_equipment" in macro_df.columns:
        biz_eq = macro_df["biz_equipment"].resample("M").last().dropna()
    else:
        # Fallback: use index-based synthetic series
        biz_eq = pd.Series(
            np.linspace(100, 120, 132),
            index=pd.date_range(start, periods=132, freq="M"),
        )

    # Compute YoY % change of the backbone
    backbone_yoy = biz_eq.pct_change(12)

    # Add noise and a cyclical amplifier (semis are ~2× more volatile)
    noise = pd.Series(
        rng.normal(0, noise_scale, len(backbone_yoy)),
        index=backbone_yoy.index,
    )
    simulated = (backbone_yoy * 2.0 + noise).dropna()

    # Align to requested window
    simulated = simulated.loc[start:end]
    simulated.name = "semi_revenue_growth_simulated"
    return simulated


# ---------------------------------------------------------------------------
# Market cap helper
# ---------------------------------------------------------------------------

def fetch_market_caps(
    tickers: list[str],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate historical market cap from prices and current shares outstanding.

    Since yfinance does not provide historical share counts, we use the
    current shares_outstanding from ``.info`` and multiply by historical
    prices.  This is a simplification — actual share counts change over time
    due to buybacks and issuances.

    Args:
        tickers: List of tickers.
        prices: DataFrame of adjusted close prices (dates × tickers).

    Returns:
        DataFrame of estimated market caps in USD (dates × tickers).
    """
    shares: dict[str, float] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            s = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            if s:
                shares[ticker] = float(s)
        except Exception:
            shares[ticker] = np.nan

    mcap_df = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for ticker in tickers:
        if ticker in prices.columns and ticker in shares:
            mcap_df[ticker] = prices[ticker] * shares[ticker]

    return mcap_df
