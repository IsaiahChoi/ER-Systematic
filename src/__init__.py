"""
src package for systematic-equity-research-semiconductors.

Exposes the main public API of each submodule so callers can do:

    from src import data_loader, factors, portfolio, analytics
    from src.factors import compute_all_factors
"""

from src import data_loader, universe, factors, portfolio, analytics, utils

__all__ = [
    "data_loader",
    "universe",
    "factors",
    "portfolio",
    "analytics",
    "utils",
]
