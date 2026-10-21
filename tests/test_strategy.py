"""Degradation fits, pit-loss measurement and plan comparison."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from f1tyre.strategy import (
    COMPOUND_ORDER,
    MIN_LAPS_FOR_FIT,
    StrategyPlan,
    degradation_curves,
    measured_pit_loss,
    simulate_plan,
    strategy_comparison,
)

ONE_STOP = StrategyPlan(name="one-stop", compounds=["MEDIUM", "HARD"], stops=[20])
TWO_STOP = StrategyPlan(
    name="two-stop", compounds=["MEDIUM", "HARD", "MEDIUM"], stops=[15, 35]
)
NO_STOP = StrategyPlan(name="no-stop", compounds=["MEDIUM"])


# ------------------------------------------------------------- stint fitting
def test_stint_degradation_finds_every_stint(session, fits: pd.DataFrame):
    assert not fits.empty
    # the race has 20 cars on one-stop or two-stop plans
    assert len(fits) >= 20
    assert set(fits["compound"]) == {"SOFT", "MEDIUM", "HARD"}
    assert fits["laps"].min() >= MIN_LAPS_FOR_FIT


def test_stint_degradation_slopes_are_plausible(session, fits: pd.DataFrame):
    # the fit itself rejects stints outside this band. Fuel burn can mask a
    # softening tyre over a long stint, so a small negative slope is a real
    # answer from the data, not a fitting error
    assert fits["slope"].between(-0.15, 0.6).all()
    # a zero-age tyre at Monza: low-fuel first stints to the backmarkers
    assert fits["base_time"].between(84.0, 89.5).all()
    # a least-squares line through a stint of at least six laps
    assert fits["r_squared"].between(0.0, 1.0).all()


def test_one_fit_per_driver_stint(session, fits: pd.DataFrame):
    grouped = fits.groupby(["driver_number", "stint_number"]).size()
    assert (grouped == 1).all()


# ------------------------------------------------------------------ curves
def test_one_curve_per_compound(session, curves):
    assert set(curves) == {"SOFT", "MEDIUM", "HARD"}
    for curve in curves.values():
        assert curve.stints >= 1
        assert curve.laps >= MIN_LAPS_FOR_FIT
        assert curve.max_age_seen >= MIN_LAPS_FOR_FIT


def test_cliff_ages_stay_inside_the_observed_data(session, curves):
    # never plan around an age the race never showed, and never below one lap
    for curve in curves.values():
        assert curve.cliff_age <= curve.max_age_seen
        assert curve.cliff_age >= 1


def test_curves_are_ordered_soft_to_hard(session, fits):
    ordered = [c.compound for c in degradation_curves(session, fits=fits)]
    assert ordered == [c for c in COMPOUND_ORDER if c in set(ordered)]


def test_curves_are_linear_in_tyre_age(session, curves):
    for curve in curves.values():
        # expressed against a zero-age tyre, so the intercept is the pace
        assert curve.time_for(0) == pytest.approx(curve.intercept)
        assert curve.time_for(20) == pytest.approx(curve.intercept + curve.slope * 20)
    # the soft tyre is the compound that sharpens as it wears; the others
    # mostly hold pace once the fuel burn is averaged out
    assert curves["SOFT"].slope > curves["MEDIUM"].slope
    assert curves["SOFT"].slope > curves["HARD"].slope


def test_pooled_slope_stays_inside_the_observed_stints(session, fits, curves):
    # pooling a compound must not invent a slope no stint in this race showed
    for compound, curve in curves.items():
        own = fits.loc[fits["compound"] == compound, "slope"]
        assert own.min() <= curve.slope <= own.max()


# --------------------------------------------------------------- pit loss
def test_pit_loss_is_measured_from_the_race(session):
    loss = measured_pit_loss(session)
    # a pit stop cannot cost nothing, and the fit rejects anything over 45s
    assert 5.0 <= loss.seconds <= 45.0
    assert loss.stops >= 3
    assert loss.usable
    assert loss.confidence in {"low", "medium", "high"}


def test_pit_loss_spread_is_not_negative(session):
    loss = measured_pit_loss(session)
    assert loss.spread >= 0.0


# ------------------------------------------------------- plan validation
def test_plan_needs_one_compound_per_stint():
    with pytest.raises(ValueError, match="compounds == stops \\+ 1"):
        StrategyPlan(name="bad", compounds=["MEDIUM"], stops=[20]).validate(53)


def test_plan_rejects_unknown_compound():
    with pytest.raises(ValueError, match="unknown compound"):
        StrategyPlan(name="bad", compounds=["MEDIUM", "INTERMEDIATE"], stops=[20]).validate(53)


def test_plan_rejects_stop_outside_the_race():
    with pytest.raises(ValueError, match="not in 1..53"):
        StrategyPlan(name="bad", compounds=["MEDIUM", "HARD"], stops=[54]).validate(53)


def test_plan_rejects_stops_that_run_backwards():
    with pytest.raises(ValueError, match="strictly increasing"):
        StrategyPlan(
            name="bad", compounds=["MEDIUM", "HARD", "SOFT"], stops=[30, 20]
        ).validate(53)


def test_plan_accepts_a_legal_one_stop():
    ONE_STOP.validate(53)
    TWO_STOP.validate(53)
    NO_STOP.validate(53)


# ---------------------------------------------------------- plan simulation
def test_simulate_plan_covers_the_race_distance(session, curves):
    loss = measured_pit_loss(session)
    result = simulate_plan(
        session, ONE_STOP, curves,
        cliffs={name: curve.cliff_age for name, curve in curves.items()},
        pit_loss=loss.seconds,
    )
    # every lap of the distance is simulated once, and the stop lap is run on
    # fresh rubber with the pit loss charged on top of it
    assert sum(result.stint_laps) == session.total_laps
    assert len(result.stint_laps) == len(ONE_STOP.compounds)
    assert result.compounds == ONE_STOP.compounds
    assert result.stops == ONE_STOP.stops


def test_simulate_plan_charges_the_pit_loss(session, curves):
    loss = measured_pit_loss(session)
    one_stop = simulate_plan(
        session, ONE_STOP, curves,
        cliffs={name: curve.cliff_age for name, curve in curves.items()},
        pit_loss=loss.seconds,
    )
    no_stop = simulate_plan(
        session, NO_STOP, curves,
        cliffs={name: curve.cliff_age for name, curve in curves.items()},
        pit_loss=loss.seconds,
    )
    assert one_stop.pit_time == pytest.approx(loss.seconds)
    assert no_stop.pit_time == 0.0


def test_a_stop_under_safety_car_is_cheaper(session, curves):
    cliffs = {name: curve.cliff_age for name, curve in curves.items()}
    loss = measured_pit_loss(session)
    green = simulate_plan(
        session, ONE_STOP, curves, cliffs, pit_loss=loss.seconds
    )
    yellow = simulate_plan(
        session,
        ONE_STOP,
        curves,
        cliffs,
        pit_loss=loss.seconds,
        safety_car_laps=(20, 30),
    )
    assert yellow.pit_time == pytest.approx(0.5 * loss.seconds)
    assert yellow.total_time < green.total_time


def test_stretching_a_tyre_past_the_cliff_costs_more(session, curves):
    # hold the cliff at ten laps so the 53-lap plan runs well past it, and
    # drop the pit loss so only the tyre term can move the answer
    tight = replace(curves["SOFT"], cliff_age=10)
    stretched = dict(curves, SOFT=tight)
    plan = StrategyPlan(name="stretch", compounds=["SOFT"], stops=[])
    linear = simulate_plan(session, plan, stretched, {"SOFT": 10}, pit_loss=0.0)
    penalised = simulate_plan(
        session, plan, stretched, {"SOFT": 10}, pit_loss=0.0, cliff_penalty=3.0
    )
    assert linear.last_tyre_age == session.total_laps
    assert penalised.total_time > linear.total_time


def test_strategy_comparison_orders_best_first(session):
    loss = measured_pit_loss(session)
    results = strategy_comparison(
        session, [ONE_STOP, TWO_STOP, NO_STOP], pit_loss=loss
    )
    assert len(results) == 3
    times = [r.total_time for r in results]
    assert times == sorted(times)
    assert all(r.fastest_lap > 0 for r in results)


def test_strategy_comparison_measures_its_own_pit_loss(session):
    results = strategy_comparison(session, [ONE_STOP, TWO_STOP])
    assert all(r.pit_time > 0 for r in results)
