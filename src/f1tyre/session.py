"""Load and normalise the downloaded OpenF1 session data.

One :class:`RaceSession` per Grand Prix: everything the dashboard needs, with
the lap table already joined to stints, compounds, tyre age and neutralisation
state. Loading is deliberately offline - the CSVs the fetch script writes are
the source of truth, so the dashboard reproduces without the API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
CACHE_DIR = DATA_DIR / "cache"

LAP_COLUMNS = [
    "session_key",
    "driver_number",
    "lap_number",
    "lap_duration",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "i1_speed",
    "i2_speed",
    "st_speed",
    "is_pit_out_lap",
    "is_pit_in_lap",
    "date_start",
]

TABLES = (
    "session",
    "drivers",
    "laps",
    "stints",
    "pit",
    "position",
    "weather",
    "race_control",
)


def _read_csv(path: Path, **kwargs: Any) -> DataFrame:
    if not path.exists():
        return DataFrame()
    return pd.read_csv(path, low_memory=False, **kwargs)


def _seconds(series: Series) -> Series:
    """Parse a ``mm:ss.SSS`` timing column into float seconds."""
    text = series.astype("string")
    minutes = text.str.extract(r"^(\d+):", expand=False)
    rest = text.str.extract(r":(\d+\.?\d*)$", expand=False)
    total = pd.to_numeric(rest, errors="coerce")
    total = total + pd.to_numeric(minutes, errors="coerce").fillna(0) * 60.0
    return pd.to_numeric(series, errors="coerce").where(total.isna(), total)


def _to_int(value: Any) -> int:
    """Coerce a possibly-missing OpenF1 value to an int (0 when absent)."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _year_of(value: Any) -> int:
    text = str(value or "")
    return int(text[:4]) if text[:4].isdigit() else 0


def _total_laps(directory: Path, session: Series) -> int:
    """How many laps this race ran.

    The OpenF1 ``sessions`` row carries ``lap_total`` only for some seasons, so
    when it is missing the lap table itself is the authority: a race that
    reached lap 57 has 57 laps whether or not the metadata says so.
    """

    total = _to_int(session.get("lap_total"))
    if total > 0:
        return total
    try:
        laps = _read_csv(directory / "laps.csv", usecols=["lap_number"])
    except ValueError:  # a laps.csv without the column
        return 0
    if laps.empty:
        return 0
    return int(pd.to_numeric(laps["lap_number"], errors="coerce").max())


def _session_label(
    slug: str, session: Series, directory: Path
) -> str:
    """A human title for one session.

    Preferred, in order: the meeting name the API knows (``2025 Australian
    Grand Prix``), then the session type and circuit from the downloaded
    metadata, and only then the directory name, because sessions are usually
    fetched by key and ``11361`` is not a title.
    """

    meeting = _meeting_name(directory)
    year = _year_of(session.get("date_start"))
    kind = str(session.get("session_name") or "").strip()
    circuit = str(session.get("circuit_short_name") or "").strip()
    if meeting:
        label = f"{year} {meeting}" if year else meeting
        if kind and kind.lower() not in {"race", "grand prix"}:
            label = f"{label} - {kind}"
        return label
    if kind:
        detail = f" - {circuit}" if circuit else ""
        return f"{year} {kind}{detail}" if year else f"{kind}{detail}"
    return _label_from_slug(slug)


def _meeting_name(directory: Path) -> str | None:
    """The Grand Prix name from the meeting row saved next to the session."""

    meeting = _read_csv(directory / "meeting.csv")
    if meeting.empty or "meeting_name" not in meeting.columns:
        return None
    name = meeting.iloc[0]["meeting_name"]
    return str(name).strip() or None if pd.notna(name) else None


def _label_from_slug(slug: str) -> str:
    """Turn a fetch-time slug such as ``2026_italian_grand_prix`` into a title."""
    parts = slug.split("_")
    if parts and parts[0].isdigit() and len(parts[0]) == 4:
        year, rest = parts[0], parts[1:]
    else:
        year, rest = "", parts
    nice = [p.capitalize() if not p.isdigit() else p for p in rest]
    return " ".join([year, *nice]) if year else " ".join(nice)


@dataclass
class RaceSession:
    """A single Grand Prix, normalised and ready for analysis."""

    session_key: int
    label: str
    slug: str
    circuit: str
    country: str
    year: int
    total_laps: int
    drivers: DataFrame
    laps: DataFrame
    stints: DataFrame
    pit: DataFrame
    position: DataFrame
    weather: DataFrame
    race_control: DataFrame
    events: Series
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ accessors
    def driver_name(self, driver_number: int | str) -> str:
        rows = self.drivers.loc[self.drivers["driver_number"] == driver_number]
        if rows.empty:
            return str(driver_number)
        for column in ("broadcast_name", "name_acronym", "full_name"):
            if column in rows.columns and pd.notna(rows.iloc[0][column]):
                return str(rows.iloc[0][column])
        return str(driver_number)

    def driver_code(self, driver_number: int | str) -> str:
        rows = self.drivers.loc[self.drivers["driver_number"] == driver_number]
        if rows.empty or "name_acronym" not in rows.columns:
            return str(driver_number)
        return str(rows.iloc[0]["name_acronym"])

    @property
    def compounds(self) -> list[str]:
        found = sorted(self.laps["compound"].dropna().unique().tolist())
        return found

    def clean_laps(self) -> DataFrame:
        """Laps that carry a legal, representative lap time."""
        laps = self.laps
        return laps.loc[
            laps["lap_duration"].notna()
            & ~laps["is_pit_out_lap"].astype(bool)
            & ~laps["is_pit_in_lap"].astype(bool)
            & (laps["lap_duration"] < 1.10 * self.reference_time)
            & (laps["lap_duration"] > 0.85 * self.reference_time)
        ].copy()

    @property
    def reference_time(self) -> float:
        """The median legal green-flag lap time of the race, in seconds."""
        if not hasattr(self, "_reference_time"):
            clean = self.laps
            clean = clean.loc[
                clean["lap_duration"].notna() & ~clean["is_pit_out_lap"].astype(bool)
            ]
            self._reference_time = float(clean["lap_duration"].median())
        return self._reference_time

    def lap_by_number(self, lap_number: int) -> DataFrame:
        """The finishing order (position at the end of a given lap)."""
        laps = self.laps
        rows = laps.loc[laps["lap_number"] == lap_number]
        return rows.sort_values("lap_duration", na_position="last")

    # ------------------------------------------------------------- builders
    @classmethod
    def from_frames(
        cls,
        session_key: int,
        label: str,
        slug: str,
        circuit: str,
        country: str,
        year: int,
        total_laps: int,
        tables: dict[str, DataFrame],
    ) -> RaceSession:
        laps = tables.get("laps", DataFrame())
        drivers = tables.get("drivers", DataFrame())
        stints = tables.get("stints", DataFrame())
        laps = _normalise_laps(laps, stints, drivers)
        events = _neutralisation_events(
            tables.get("race_control", DataFrame()),
            tables.get("session", DataFrame()),
            total_laps,
        )
        laps = _flag_neutralisation(laps, events)
        return cls(
            session_key=session_key,
            label=label,
            slug=slug,
            circuit=circuit,
            country=country,
            year=year,
            total_laps=total_laps,
            drivers=drivers,
            laps=laps,
            stints=stints,
            pit=tables.get("pit", DataFrame()),
            position=tables.get("position", DataFrame()),
            weather=tables.get("weather", DataFrame()),
            race_control=tables.get("race_control", DataFrame()),
            events=events,
            metadata={"tables": sorted(t for t in tables if not tables[t].empty)},
        )


def _normalise_laps(
    laps: DataFrame, stints: DataFrame, drivers: DataFrame
) -> DataFrame:
    """Join lap times to stints, compounds, tyre age and driver names."""
    if laps.empty:
        return laps.assign(
            compound=Series(dtype="object"),
            tyre_age=Series(dtype="float64"),
            stint_number=Series(dtype="float64"),
            lap_seconds=Series(dtype="float64"),
            code=Series(dtype="object"),
            driver_name=Series(dtype="object"),
        )
    laps = laps.copy()
    for column in LAP_COLUMNS:
        if column not in laps.columns:
            laps[column] = pd.NA
    laps["lap_seconds"] = _seconds(laps["lap_duration"])
    for sector in (1, 2, 3):
        laps[f"sector_{sector}"] = _seconds(laps[f"duration_sector_{sector}"])

    keep = [c for c in LAP_COLUMNS if c != "date_start"]
    laps = laps[keep + ["lap_seconds", "sector_1", "sector_2", "sector_3"]]

    # Stint metadata: compound + the age the tyre reaches on each lap.
    if not stints.empty:
        stints = stints.copy()
        stints["lap_start"] = pd.to_numeric(stints["lap_start"], errors="coerce")
        stints["lap_end"] = pd.to_numeric(stints["lap_end"], errors="coerce")
        stints = stints.rename(
            columns={
                "compound": "compound",
                "lap_start": "stint_start",
                "lap_end": "stint_end",
            }
        )
        merge = stints[
            [
                c
                for c in (
                    "session_key",
                    "driver_number",
                    "stint_number",
                    "stint_start",
                    "stint_end",
                    "compound",
                )
                if c in stints.columns
            ]
        ]
        laps = laps.merge(merge, on=["session_key", "driver_number"], how="left")
    else:
        laps["stint_number"] = pd.NA
        laps["stint_start"] = pd.NA
        laps["stint_end"] = pd.NA
        laps["compound"] = pd.NA

    laps = laps.loc[
        laps["lap_number"] >= laps["stint_start"].fillna(0)
    ]
    laps = laps.loc[
        laps["lap_number"] <= laps["stint_end"].fillna(laps["lap_number"])
    ]
    laps["tyre_age"] = laps["lap_number"] - laps["stint_start"] + 1
    laps["stint_number"] = laps["stint_number"].astype("Float64")
    laps["tyre_age"] = laps["tyre_age"].astype("Float64")

    if not drivers.empty:
        names = drivers[["driver_number", "name_acronym", "broadcast_name"]].copy()
        names = names.rename(
            columns={"name_acronym": "code", "broadcast_name": "driver_name"}
        )
        laps = laps.merge(names, on="driver_number", how="left")
    else:
        laps["code"] = laps["driver_number"].astype(str)
        laps["driver_name"] = laps["code"]

    for flag in ("is_pit_out_lap", "is_pit_in_lap"):
        # OpenF1 omits some of these flags entirely, and the ones it does send
        # arrive as real booleans, so coerce through the nullable boolean
        # dtype: an object column of ``pd.NA`` would otherwise have to be
        # downcast by ``fillna`` to reach a usable dtype.
        laps[flag] = laps[flag].astype("boolean").fillna(False).astype(bool)
    return laps.reset_index(drop=True)

_SC_START = re.compile(r"\b(?:VIRTUAL )?SAFETY CAR DEPLOYED\b")
_SC_END = re.compile(r"\b(?:VIRTUAL )?SAFETY CAR (?:ENDING|IN THIS LAP)\b")
_RED_START = re.compile(r"RED FLAG")
_RED_END = re.compile(r"RESUMPTION ORDER|RACE WILL RESUME|RACE RESUMED|RED FLAG CLEARED")


def _neutralisation_events(
    race_control: DataFrame, session: DataFrame, total_laps: int | None = None
) -> Series:
    """A per-lap boolean: was a safety car / VSC / red flag in force?"""
    empty = Series(False)
    if race_control.empty or "lap_number" not in race_control.columns:
        return empty
    events = race_control.copy()
    events["message"] = events["message"].astype("string").str.upper().fillna("")
    events["lap_number"] = pd.to_numeric(events["lap_number"], errors="coerce")
    events = events.loc[events["lap_number"].notna()]
    if events.empty:
        return empty

    if not total_laps:
        if session.empty:
            total_laps = int(events["lap_number"].max() or 0)
        else:
            total_laps = _to_int(session.iloc[0].get("lap_total"))
        if total_laps <= 0:
            total_laps = int(events["lap_number"].max())

    windows: list[tuple[int, int]] = []
    open_lap: int | None = None
    for lap, message in zip(events["lap_number"].astype(int), events["message"], strict=False):
        if _SC_START.search(message) or _RED_START.search(message):
            open_lap = lap if open_lap is None else min(open_lap, lap)
        elif open_lap is not None and (
            _SC_END.search(message) or _RED_END.search(message)
        ):
            # The neutralisation runs up to the lap *before* it clears.
            windows.append((open_lap, max(lap - 1, open_lap)))
            open_lap = None
    if open_lap is not None:
        windows.append((open_lap, total_laps))

    per_lap = Series(
        [
            any(start <= lap <= end for start, end in windows)
            for lap in range(total_laps + 1)
        ],
        name="neutralised",
        dtype=bool,
    )
    return per_lap


def _flag_neutralisation(laps: DataFrame, events: Series) -> DataFrame:
    if events.empty or "lap_number" not in laps.columns:
        laps["neutralised"] = False
    else:
        index = laps["lap_number"].astype(int).clip(0, len(events) - 1)
        laps["neutralised"] = events.reindex(index).to_numpy()
    return laps


class SessionCatalogue:
    """Discover the races that have been downloaded to ``data/sessions``."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else SESSIONS_DIR
        self.index = self._build_index()

    def _build_index(self) -> list[dict[str, Any]]:
        catalogue: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("*/session.csv")):
            key = path.parent.name
            try:
                row = pd.read_csv(path)
            except Exception:
                continue
            if row.empty:
                continue
            record = row.iloc[0]
            catalogue.append(
                {
                    "session_key": int(record.get("session_key", key)),
                    "slug": path.parent.name,
                    "path": path.parent,
                    "circuit": record.get("circuit_short_name") or "Unknown circuit",
                    "country": record.get("country_name") or record.get("location"),
                "year": _year_of(record.get("date_start")),
                "total_laps": _total_laps(path.parent, record),
                "label": _session_label(path.parent.name, record, path.parent),
                }
            )
        return catalogue

    def __iter__(self):
        return iter(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def default(self) -> int | None:
        """The session key to open by default: the newest complete race."""
        if not self.index:
            return None
        return max(self.index, key=lambda r: (r["year"], r["session_key"]))[
            "session_key"
        ]


def _read_tables(directory: Path) -> dict[str, DataFrame]:
    tables: dict[str, DataFrame] = {}
    for name in TABLES:
        path = directory / f"{name}.csv"
        tables[name] = _read_csv(path)
    return tables


def load_session(session_key: int, catalogue: SessionCatalogue | None = None) -> RaceSession:
    """Load one race by its OpenF1 session key."""
    catalogue = catalogue or SessionCatalogue()
    record = next(
        (r for r in catalogue if r["session_key"] == int(session_key)), None
    )
    if record is None:
        raise KeyError(f"session {session_key} has not been downloaded")
    tables = _read_tables(record["path"])
    return RaceSession.from_frames(
        session_key=record["session_key"],
        label=record["label"],
        slug=record["slug"],
        circuit=record["circuit"],
        country=record["country"],
        year=record["year"],
        total_laps=record["total_laps"],
        tables=tables,
    )


def load_sessions(keys: list[int] | None = None) -> list[RaceSession]:
    """Load every downloaded race, or the requested subset."""
    catalogue = SessionCatalogue()
    keys = keys if keys else [record["session_key"] for record in catalogue]
    return [load_session(key, catalogue) for key in keys]

