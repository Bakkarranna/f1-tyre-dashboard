"""Fixtures loading the committed race extracts.

The three races under ``data/sessions`` are the project's test data: they are
small enough to read in under a second and they carry every case the analysis
code has to cope with - three compounds, stints from one to four per driver,
Safety Car laps, and two seasons of the same circuit for contrast.
"""

from __future__ import annotations

import pytest

from f1tyre.session import SessionCatalogue, load_session, load_sessions
from f1tyre.strategy import degradation_curves, stint_degradation

#: 2026 Italian Grand Prix at Monza: 53 laps, three compounds, 22 drivers.
MONZA_2026 = 11361


@pytest.fixture(scope="session")
def catalogue() -> SessionCatalogue:
    return SessionCatalogue()


@pytest.fixture(scope="session")
def session():
    return load_session(MONZA_2026)


@pytest.fixture(scope="session")
def sessions(catalogue: SessionCatalogue) -> list:
    return load_sessions()


@pytest.fixture(scope="session")
def fits(session):
    return stint_degradation(session)


@pytest.fixture(scope="session")
def curves(session, fits) -> dict:
    return {c.compound: c for c in degradation_curves(session, fits=fits)}
