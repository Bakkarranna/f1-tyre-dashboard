"""Tyre performance and degradation analysis for a completed Grand Prix."""

from .session import RaceSession, SessionCatalogue, load_session, load_sessions
from .strategy import (
    StrategyResult,
    degradation_curves,
    stint_degradation,
    strategy_comparison,
)

__all__ = [
    "RaceSession",
    "SessionCatalogue",
    "load_session",
    "load_sessions",
    "StrategyResult",
    "degradation_curves",
    "stint_degradation",
    "strategy_comparison",
]

__version__ = "1.0.0"
