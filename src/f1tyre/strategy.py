"""Degradation estimation and strategy comparison.

Two jobs:

* :func:`degradation_curves` - how much lap time a tyre loses per lap of age,
  from the laps that were actually completed. Stints are pooled by compound and
  each one's own laps are centred on that stint's pace first, so fuel burn and
  circuit evolution do not masquerade as degradation.
* :func:`strategy_comparison` - what a strategy *would* have cost, by replaying
  a stint plan through those curves plus the pit-lane time loss measured in the
  same session. This is an estimate with stated assumptions, not a simulation
  of the race that happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from .session import RaceSession

COMPOUND_ORDER = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
MIN_LAPS_FOR_FIT = 6
CLIFF_TOLERANCE = 0.75
CLIFF_PENALTY = 2.0


@dataclass
class StintFit:
    """The fitted pace of one driver's one stint."""

    driver_number: int
    code: str
    stint_number: float
    compound: str
    first_lap: int
    last_lap: int
    laps: int
    base_time: float  # seconds, at the tyre's first racing lap
    slope: float  # seconds lost per lap of tyre age
    r_squared: float
    mean_tyre_age: float
    neutralised_laps: int

    @property
    def lap_time_at(self) -> Any:  # pragma: no cover - convenience accessor
        return lambda age: self.base_time + self.slope * age


@dataclass
class CompoundCurve:
    """Aggregated degradation for one compound, pooled across stints."""

    compound: str
    stints: int
    laps: int
    intercept: float  # pace of a tyre with zero age
    slope: float  # s/lap of tyre age
    slope_se: float  # standard error of the slope
    r_squared: float
    cliff_age: float  # age where the linear fit crosses the observed limit
    max_age_seen: int

    def time_for(self, tyre_age: float) -> float:
        """Predicted lap time for a tyre of this age, in seconds."""
        return self.intercept + self.slope * tyre_age


def _fit_line(ages: Series, times: Series) -> tuple[float, float, float]:
    """Ordinary least squares of lap time on tyre age.

    Returns ``(intercept, slope, r_squared)``. A plain fit is deliberate: with
    ten to fifty laps per stint, a quadratic buys an extra parameter and mostly
    fits noise, and the residual structure is easier to explain to a race
    engineer than the curve that happens to leave one unexplained lap.
    """
    ages = ages.astype(float)
    times = times.astype(float)
    slope, intercept = np.polyfit(ages, times, 1)
    predicted = intercept + slope * ages
    residual = float(((times - predicted) ** 2).sum())
    total = float(((times - times.mean()) ** 2).sum())
    r_squared = 1.0 - residual / total if total > 1e-9 else 0.0
    return float(intercept), float(slope), r_squared


def stint_degradation(
    session: RaceSession, *, min_laps: int = MIN_LAPS_FOR_FIT
) -> DataFrame:
    """Fit one degradation line per stint.

    Every stint becomes ``lap_time = intercept + slope x tyre_age``. Stints
    that are too short to fit, or whose fitted slope is not remotely plausible
    for a tyre (negative, or worse than half a second per lap), are dropped
    rather than dragging the compound average around.
    """
    laps = session.clean_laps()
    if laps.empty:
        return DataFrame()

    rows: list[dict] = []
    for (driver, stint), group in laps.groupby(["driver_number", "stint_number"]):
        if stint is pd.NA or len(group) < min_laps:
            continue
        group = group.sort_values("lap_number")
        ages = group["tyre_age"].astype(float)
        times = group["lap_seconds"].astype(float)
        intercept, slope, r_squared = _fit_line(ages, times)
        if not -0.15 <= slope <= 0.60:
            continue
        rows.append(
            {
                "session_key": session.session_key,
                "driver_number": driver,
                "code": group["code"].iloc[0],
                "stint_number": float(stint),
                "compound": group["compound"].iloc[0],
                "first_lap": int(group["lap_number"].min()),
                "last_lap": int(group["lap_number"].max()),
                "laps": int(len(group)),
                "base_time": intercept,
                "slope": slope,
                "r_squared": r_squared,
                "mean_tyre_age": float(ages.mean()),
                "neutralised_laps": int(group["neutralised"].sum()),
            }
        )
    frame = DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["driver_number", "first_lap"]).reset_index(drop=True)


def _cliff_from_pool(
    pool: DataFrame, intercept: float, slope: float, fallback: int
) -> int:
    """The age where the observed laps leave the fitted line (see above)."""

    if pool.empty:
        return int(fallback)
    binned = pool.groupby("tyre_age")["lap_seconds"].agg(["count", "mean"])
    binned = binned[binned["count"] >= 5]
    for age, row in binned.sort_index().iterrows():
        if row["mean"] > intercept + slope * age + CLIFF_TOLERANCE:
            return int(age)
    return int(fallback)


def _cliff_age(
    session: RaceSession, curve: CompoundCurve, starts: DataFrame
) -> int:
    """The tyre age after which the compound stops worsening linearly.

    Laps are binned by age; once the observed mean in a bin sits more than
    :data:`CLIFF_TOLERANCE` above the fitted line, on at least five laps, the
    tyre has left the linear regime - the age a strategist stops planning
    around. Returns ``max_age_seen`` when every bin still sits on the line,
    which is the honest "no cliff inside this race's data" answer.
    """
    pool = _pool(
        session,
        starts,
        curve.compound,
        MIN_LAPS_FOR_FIT,
        curve.max_age_seen,
    )
    return _cliff_from_pool(
        pool, curve.intercept, curve.slope, curve.max_age_seen
    )


def degradation_curves(
    session: RaceSession,
    *,
    min_laps: int = MIN_LAPS_FOR_FIT,
    fits: DataFrame | None = None,
) -> list[CompoundCurve]:
    """One degradation model per compound, pooled across every stint.

    The slope is pooled with stint length as the weight; the base pace is the
    median of its stints' intercepts, which keeps one quick car from setting
    the field's pace. Every curve is expressed relative to a zero-age tyre, so
    a strategy simulation can call ``curve.time_for(age)`` for any stint length
    without knowing which driver it belongs to.
    """
    frame = stint_degradation(session, min_laps=min_laps) if fits is None else fits
    if frame.empty:
        return []

    curves: list[CompoundCurve] = []
    for compound, group in frame.groupby("compound"):
        weights = group["laps"].astype(float).to_numpy()
        slopes = group["slope"].to_numpy()
        bases = group["base_time"].to_numpy()
        slope = float(np.average(slopes, weights=weights))
        base = float(np.median(bases))
        residual = weights * (slopes - slope) ** 2
        dof = max(len(slopes) - 1, 1)
        slope_se = float(np.sqrt((residual.sum() / dof) / (weights.sum() / dof)))
        weighted_r2 = float(np.average(group["r_squared"], weights=weights))
        max_age = int(session.laps.loc[session.laps["compound"] == compound, "tyre_age"].max())
        curve = CompoundCurve(
            compound=str(compound),
            stints=int(len(group)),
            laps=int(weights.sum()),
            intercept=base,
            slope=slope,
            slope_se=slope_se,
            r_squared=weighted_r2,
            cliff_age=max_age,
            max_age_seen=max_age,
        )
        curves.append(curve)
    curves.sort(key=lambda c: COMPOUND_ORDER.get(c.compound, 9))
    return curves


# --------------------------------------------------------------- pace pool
PACE_COLUMNS = [
    "driver_number",
    "lap_number",
    "stint_number",
    "compound",
    "tyre_age",
    "lap_seconds",
    "neutralised",
]


def _pool(
    session: RaceSession,
    starts: DataFrame,
    compound: str,
    min_laps: int,
    max_age: int,
) -> DataFrame:
    """One compound's green-flag laps as deltas against each driver's own pace.

    A driver who is three seconds a lap slower than the leader contributes to
    the pool only if they are slower than *themselves*, which is what the slope
    is meant to capture.
    """
    laps = session.clean_laps()
    laps = laps.loc[
        (laps["compound"] == compound)
        & laps["lap_seconds"].notna()
        & ~laps["neutralised"].astype(bool)
    ]
    if starts.empty:
        return DataFrame(columns=PACE_COLUMNS + ["delta"])
    joined = laps.merge(
        starts[["driver_number", "stint_number", "base_time"]],
        on=["driver_number", "stint_number"],
        how="inner",
    )
    if joined.empty:
        return DataFrame(columns=PACE_COLUMNS + ["delta"])
    joined = joined.copy()
    joined["delta"] = joined["lap_seconds"] - joined["base_time"]
    pool = joined.loc[joined["delta"].abs() <= 3.0]
    if len(pool) < min_laps:
        return DataFrame(columns=PACE_COLUMNS + ["delta"])
    pool = pool.loc[pool["tyre_age"] <= max_age]
    return pool[PACE_COLUMNS + ["delta"]].reset_index(drop=True)


# ---------------------------------------------------------------- pit loss
@dataclass
class PitLoss:
    """The time cost of one pit stop, measured in this session."

    ``seconds`` is the *total* time lost across the lap that stops and the lap
    after it. Strategy simulations charge that whole amount to the stop lap,
    which is why it is not halved before use.
    """

    seconds: float
    stops: int
    spread: float
    confidence: str

    @property
    def usable(self) -> bool:
        return self.stops >= 3


def _pace_between(
    laps: DataFrame, driver: int, low: int, high: int
) -> float | None:
    window = laps.loc[
        (laps["driver_number"] == driver)
        & (laps["lap_number"] >= low)
        & (laps["lap_number"] <= high)
        & ~laps["neutralised"]
        & laps["lap_seconds"].notna()
        & ~laps["is_pit_in_lap"].astype(bool)
        & ~laps["is_pit_out_lap"].astype(bool)
    ]
    if window.empty:
        return None
    return float(window["lap_seconds"].median())


def measured_pit_loss(
    session: RaceSession, *, min_pace_laps: int = 5, max_loss: float = 45.0
) -> PitLoss:
    """Measure the pit-lane time loss from the stops this race actually saw.

    The stints table says *when* a stop happened; the lap times say what it
    cost. For each stop the in-lap and out-lap are compared against the median
    pace the same driver was running either side of the stop, so a slow pit
    lane or a Safety Car stop is not averaged into the field's track position.
    """
    stops: list[float] = []
    laps = session.laps
    for (driver, stint), group in laps.groupby(["driver_number", "stint_number"]):
        if pd.isna(stint) or stint == 1:
            continue
        group = group.sort_values("lap_number")
        out_lap = int(group.iloc[0]["lap_number"])
        in_lap = out_lap - 1
        if in_lap < 1 or out_lap > session.total_laps:
            continue
        before = _pace_between(laps, driver, in_lap - min_pace_laps, in_lap - 1)
        after = _pace_between(laps, driver, out_lap + 2, out_lap + 1 + min_pace_laps)
        in_time = float(laps.loc[laps["lap_number"] == in_lap].iloc[0]["lap_seconds"])
        out_time = float(laps.loc[laps["lap_number"] == out_lap].iloc[0]["lap_seconds"])
        if before is None or after is None:
            continue
        if in_lap in session.laps.loc[session.laps["neutralised"], "lap_number"].values:
            continue
        loss = (in_time - before) + (out_time - after)
        if 5.0 <= loss <= max_loss:
            stops.append(loss)
    if not stops:
        return PitLoss(seconds=0.0, stops=0, spread=0.0, confidence="none")
    if len(stops) >= 8:
        confidence = "high"
    elif len(stops) >= 4:
        confidence = "medium"
    else:
        confidence = "low"
    return PitLoss(
        seconds=float(np.median(stops)),
        stops=int(len(stops)),
        spread=float(np.percentile(stops, 90) - np.percentile(stops, 10)),
        confidence=confidence,
    )


# --------------------------------------------------------- strategy engine
@dataclass
class StrategyPlan:
    """A stint order and the laps the stops are taken on."

    ``compounds`` has one entry per stint, oldest first, so a MEDIUM then HARD
    one-stop is ``["MEDIUM", "HARD"]`` with ``stops=[15]``.
    """

    name: str
    compounds: list[str]
    stops: list[int] = field(default_factory=list)

    def validate(self, total_laps: int) -> None:
        if len(self.compounds) != len(self.stops) + 1:
            raise ValueError(
                f"{self.name}: {len(self.compounds)} compounds but "
                f"{len(self.stops)} stops (need compounds == stops + 1)"
            )
        if self.compounds != self.compounds[:1] * (len(self.stops) + 1):
            if not set(self.compounds).issubset(COMPOUND_ORDER):
                raise ValueError(f"{self.name}: unknown compound in {self.compounds}")
        for stop in self.stops:
            if not 1 <= stop <= total_laps:
                raise ValueError(f"{self.name}: stop on lap {stop} not in 1..{total_laps}")
        if any(b <= a for a, b in zip(self.stops, self.stops[1:], strict=False)):
            raise ValueError(f"{self.name}: stops must be strictly increasing")


@dataclass
class StrategyResult:
    """The estimated outcome of running one plan."""

    name: str
    compounds: list[str]
    stops: list[int]
    total_time: float
    stint_laps: list[int]
    pit_time: float
    fastest_lap: float
    last_tyre_age: int


def _lap_time(
    curve: CompoundCurve, age: int, cliff: int, penalty: float
) -> float:
    lap = curve.intercept + curve.slope * min(age, cliff)
    if age > cliff:
        lap += curve.slope * penalty * (age - cliff)
    return lap


def simulate_plan(
    session: RaceSession,
    plan: StrategyPlan,
    curves: dict[str, CompoundCurve],
    cliffs: dict[str, int],
    *,
    pit_loss: float,
    total_laps: int | None = None,
    cliff_penalty: float = CLIFF_PENALTY,
    safety_car_laps: tuple[int, int] | None = None,
    safety_car_factor: float = 0.5,
) -> StrategyResult:
    """Replay one stint plan through the fitted degradation curves.

    Every assumption is on the surface of the call: the pit loss is charged in
    full to the stop lap, and past the cliff age the modelled degradation is
    multiplied by ``cliff_penalty`` so a plan that stretches a tyre is not
    flattered by the linear fit it was fitted on.
    """
    total_laps = total_laps or session.total_laps
    plan.validate(total_laps)
    stop_set = set(plan.stops)
    times: list[float] = []
    stint_laps: list[int] = []
    stint = 0
    age = 0
    for lap in range(1, total_laps + 1):
        if lap in stop_set:
            stint += 1
            age = 0
            if safety_car_laps and safety_car_laps[0] <= lap <= safety_car_laps[1]:
                stint_laps.append(0)
                times.append(pit_loss * safety_car_factor)
                continue
        age += 1
        compound = plan.compounds[stint]
        curve = curves[compound]
        times.append(_lap_time(curve, age, cliffs[compound], cliff_penalty))
        if stint == len(stint_laps):
            stint_laps.append(1)
        else:
            stint_laps[stint] += 1
    # the pit loss is charged to the stop lap on top of the fresh-tyre lap that
    # lap also runs, so total_time counts every lap of the distance exactly once
    pit_time = sum(
        pit_loss * (
            1.0
            if not (safety_car_laps and safety_car_laps[0] <= s <= safety_car_laps[1])
            else safety_car_factor
        )
        for s in stop_set
    )
    return StrategyResult(
        name=plan.name,
        compounds=list(plan.compounds),
        stops=list(plan.stops),
        total_time=float(sum(times)) + pit_time,
        stint_laps=stint_laps,
        pit_time=float(pit_time),
        fastest_lap=float(min(times)) if times else 0.0,
        last_tyre_age=age,
    )


def strategy_comparison(
    session: RaceSession,
    plans: list[StrategyPlan],
    *,
    fits: DataFrame | None = None,
    total_laps: int | None = None,
    pit_loss: PitLoss | None = None,
    safety_car_laps: tuple[int, int] | None = None,
) -> list[StrategyResult]:
    """Estimate the race time of each plan on this race's own numbers.

    Curves come from the green-flag stints actually run; the pit loss is the
    one measured from the stops actually made. Results are ordered best first.
    """
    total_laps = total_laps or session.total_laps
    loss = pit_loss or measured_pit_loss(session)
    if not loss.usable:
        raise ValueError(
            "cannot estimate strategy: fewer than 4 informative stops in this "
            "session (pit stops happened under a Safety Car or the stint data "
            "is incomplete)"
        )
    frames = stint_degradation(session) if fits is None else fits
    curves = {c.compound: c for c in degradation_curves(session, fits=frames)}
    if not curves:
        raise ValueError("cannot estimate strategy: no degradation curves fitted")
    cliffs = {
        c.compound: _cliff_age(session, c, frames) for c in curves.values()
    }
    results = [
        simulate_plan(
            session,
            plan,
            curves,
            cliffs,
            pit_loss=loss.seconds,
            total_laps=total_laps,
            safety_car_laps=safety_car_laps,
        )
        for plan in plans
    ]
    results.sort(key=lambda r: r.total_time)
    return results

