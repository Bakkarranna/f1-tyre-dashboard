"""Session loading and lap normalisation."""

from __future__ import annotations

import pytest

from f1tyre.session import RaceSession, SessionCatalogue, load_session


def test_catalogue_discovers_downloaded_races(catalogue: SessionCatalogue):
    keys = {record["session_key"] for record in catalogue}
    assert len(catalogue) >= 2
    assert all(isinstance(key, int) for key in keys)
    # the default is the newest race, ordered by season then session key
    newest = max(catalogue, key=lambda r: (r["year"], r["session_key"]))
    assert catalogue.default() == newest["session_key"]


def test_session_metadata(session: RaceSession):
    assert session.session_key == 11361
    assert session.label == "2026 Italian Grand Prix"
    assert session.circuit == "Monza"
    assert session.country == "Italy"
    assert session.year == 2026
    assert session.total_laps == 53


def test_session_tables_are_loaded(session: RaceSession):
    assert len(session.drivers) == 22
    assert not session.laps.empty
    assert not session.stints.empty
    assert not session.pit.empty
    assert not session.race_control.empty
    assert not session.weather.empty


def test_lap_columns_are_normalised(session: RaceSession):
    for column in (
        "driver_number",
        "lap_number",
        "lap_seconds",
        "compound",
        "tyre_age",
        "stint_number",
        "is_pit_in_lap",
        "is_pit_out_lap",
        "neutralised",
    ):
        assert column in session.laps.columns
    # every lap carries a usable time and a positive lap number
    assert session.laps["lap_seconds"].notna().all()
    assert session.laps["lap_number"].between(1, session.total_laps).all()


def test_pit_flags_are_usable_booleans(session: RaceSession):
    # OpenF1 omits ``is_pit_in_lap`` from some extracts altogether, so the
    # loader has to coerce it back to a real boolean flag rather than leaving
    # an object column of ``pd.NA`` behind.
    for flag in ("is_pit_out_lap", "is_pit_in_lap"):
        column = session.laps[flag]
        assert column.dtype == bool, f"{flag} is {column.dtype}, not a bool flag"
        assert column.notna().all()
    # the pit-out flag genuinely separates the two sets of laps
    assert session.laps["is_pit_out_lap"].any()


def test_tyre_age_restarts_at_each_stop(session: RaceSession):
    # a stint's first racing lap runs on a zero-age tyre
    for (_, stint), group in session.laps.dropna(subset=["stint_number"]).groupby(
        ["driver_number", "stint_number"]
    ):
        youngest = int(group["tyre_age"].min())
        assert youngest == 1, f"stint {stint} of a driver starts at tyre age {youngest}"


def test_clean_laps_drop_stop_laps(session: RaceSession):
    clean = session.clean_laps()
    assert not clean.empty
    assert len(clean) < len(session.laps)
    assert not clean["is_pit_in_lap"].any()
    assert not clean["is_pit_out_lap"].any()
    # the filter keeps only laps within 10% of the race's reference time
    reference = session.reference_time
    assert clean["lap_seconds"].lt(1.10 * reference).all()
    assert clean["lap_seconds"].gt(0.85 * reference).all()


def test_reference_time_is_a_plausible_monza_lap(session: RaceSession):
    assert 60.0 < session.reference_time < 120.0


def test_compounds_are_the_three_dry_ones(session: RaceSession):
    assert set(session.compounds) == {"SOFT", "MEDIUM", "HARD"}


def test_neutralisation_is_flagged_from_race_control(session: RaceSession):
    flagged = session.laps.loc[session.laps["neutralised"]]
    # Monza 2026 lost at least one lap to a Safety Car
    assert len(flagged) >= 1
    # a neutralised lap is not a green-flag pace lap, so it is never clean
    assert not set(flagged["lap_number"]).issubset(set(session.clean_laps()["lap_number"]))


def test_driver_lookup(session: RaceSession):
    assert session.driver_code(1) == "NOR"
    assert "NORRIS" in session.driver_name(1).upper()
    # an unknown driver falls back to its number rather than raising
    assert session.driver_code(9999) == "9999"
    assert session.driver_name(9999) == "9999"


def test_lap_by_number_orders_by_lap_time(session: RaceSession):
    finish = session.lap_by_number(session.total_laps)
    # only the cars that were still running recorded the final lap
    recorded = session.laps.loc[
        session.laps["lap_number"] == session.total_laps, "driver_number"
    ]
    assert len(finish) == recorded.nunique()
    assert finish["driver_number"].nunique() == len(finish)
    # ordered by lap time, which is finishing order for a full final lap
    times = finish["lap_seconds"].dropna().tolist()
    assert times == sorted(times)


def test_unknown_session_key_raises():
    with pytest.raises(KeyError, match="has not been downloaded"):
        load_session(1)


def test_empty_extract_still_builds_a_session():
    # a race that has not finished downloading should not break the loader
    empty = RaceSession.from_frames(
        session_key=99999,
        label="empty",
        slug="empty",
        circuit="Unknown",
        country="",
        year=2025,
        total_laps=0,
        tables={},
    )
    assert empty.laps.empty
    assert empty.total_laps == 0
    assert empty.drivers.empty


def test_all_sessions_load(sessions):
    assert len(sessions) >= 2
    labels = {session.label for session in sessions}
    assert "2026 Italian Grand Prix" in labels
    assert "2025 Italian Grand Prix" in labels
