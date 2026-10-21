"""Download the dataset for the tyre-performance dashboard.

Usage::

    python scripts/fetch_openf1.py                 # default session list
    python scripts/fetch_openf1.py --telemetry   # also pull 4 Hz car_data
    python scripts/fetch_openf1.py --refresh       # ignore the local cache

One folder per session is written under ``data/sessions/<key>``:

* ``laps.csv``           - one row per lap, joined with the stint/compound data
* ``drivers.csv``        - driver number to name/team mapping
* ``stints.csv``         - stint records straight from the API
* ``race_control.csv``   - flags, safety-car and other race-control messages
* ``weather.csv``        - per-sample air/track temperature and rainfall
* ``pit.csv``            - pit stop events with pit-lane time loss
* ``position.csv``       - classified position at each point in the session
* ``session.csv``        - session metadata (year, meeting, circuit, type)
* ``telemetry.csv.gz``   - 4 Hz speed/throttle/brake/n_gear/DRS for the
                           representative lap of every driver (opt out with
                           ``--no-telemetry``)

Responses are cached under ``data/cache/openf1/`` so a re-run is cheap and the
API is not hammered; delete that folder (or pass ``--refresh``) to re-download.

Requires the ``requests`` package (installed by ``pip install -e ".[fetch]"``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.openf1.org/v1"
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache" / "openf1"

# Three races, each chosen for a job in the dashboard:
#
# 11361 = 2026 Italian GP (Monza)  - the headline session: all three dry
#         compounds were used, so Soft / Medium / Hard can be compared on one
#         weekend's data
# 11334 = 2026 Belgian GP (Spa)    - a high-load circuit the same season, to
#         check whether the degradation ranking is the circuit's or general
# 9912  = 2025 Italian GP (Monza)  - the same circuit a year earlier, to
#         separate circuit behaviour from that particular race's weather
SESSION_KEYS = [11361, 11334, 9912]

TIMEOUT = 60
MIN_DELAY = 0.4  # seconds between API calls, to stay inside the rate limit


def _get(path: str, *, refresh: bool = False, **params) -> list[dict]:
    """GET one OpenF1 endpoint and return the decoded rows.

    Responses are cached on disk keyed by the endpoint and its parameters, so
    repeated runs of this script (and of the tests) do not re-hit the network.
    """
    url = f"{BASE}/{path}"
    key = hashlib.sha1(
        json.dumps([path, sorted(params.items())], sort_keys=True).encode()
    ).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{path}_{key}.json"

    if not refresh and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            pass  # a truncated cache file is simply ignored

    for attempt in range(4):
        try:
            time.sleep(MIN_DELAY)
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            if resp.status_code == 404:
                # OpenF1 answers 404 with {"detail": "No results found."} - an
                # empty result, not an error (car_data is a common case).
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text("[]", encoding="utf-8")
                return []
            if resp.status_code == 429 or resp.status_code >= 500:
                continue  # rate limited or server error: back off and retry
            resp.raise_for_status()
            payload = resp.json()
            rows = payload if isinstance(payload, list) else []
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(rows), encoding="utf-8")
            return rows
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            continue
    return []


def _fetch_all_pages(path: str, **params) -> list[dict]:
    """Page through an endpoint until fewer rows than the limit come back."""
    limit = 1000
    rows: list[dict] = []
    offset = 0
    while True:
        page = _get(path, limit=limit, offset=offset, **params)
        rows.extend(page)
        if len(page) < limit:
            return rows
        offset += limit


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows to a DataFrame that is sorted, de-duplicated and json-serialisable."""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    # OpenF1 nests some fields (segment arrays, sector lists). Flatten them to
    # JSON text first so sorting and de-duplication cannot trip over lists.
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (list, dict))).any():
            frame[column] = frame[column].map(json.dumps)
    if "date" in frame.columns:
        frame = frame.sort_values("date", kind="stable")
    frame = frame.drop_duplicates()
    return frame


def _write_csv(frame: pd.DataFrame, path: Path) -> int:
    """Write a CSV (compressed when the name ends in .gz) and return its rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
    else:
        frame.to_csv(path, index=False)
    return len(frame)


def _sessions_detail(session_key: int) -> list[dict]:
    rows = _get("sessions", session_key=session_key)
    if rows:
        return rows
    # Some sessions only resolve through the year-scoped listing.
    return _fetch_all_pages("sessions")


def _attach_stints(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Attach compound and tyre age to every lap from the stint records.

    The API calls the stint column ``stint_number`` and records the lap span each
    stint ran for, so laps and stints are joined on a range (``lap_start`` to
    ``lap_end`` per driver) rather than on a key. Joining on driver number alone
    would fan every lap out over all of that driver's stints.

    Returns the lap frame with ``stint``, ``compound``, ``tyre_age_at_start`` and
    ``tyre_age`` columns added.
    """
    if laps.empty:
        return laps.copy()

    for column in ("stint", "compound", "tyre_age_at_start", "tyre_age"):
        laps[column] = pd.NA

    if stints.empty:
        return laps

    if "stint_number" in stints.columns and "stint" not in stints.columns:
        stints = stints.rename(columns={"stint_number": "stint"})

    for record in stints.to_dict("records"):
        driver = record.get("driver_number")
        stint = record.get("stint")
        if pd.isna(driver) or pd.isna(stint):
            continue
        lap_start = record.get("lap_start")
        lap_end = record.get("lap_end")
        if pd.isna(lap_start) or pd.isna(lap_end):
            continue
        in_stint = (
            (laps["driver_number"] == driver)
            & (laps["lap_number"] >= int(lap_start))
            & (laps["lap_number"] <= int(lap_end))
        )
        age = record.get("tyre_age_at_start")
        age = 0 if pd.isna(age) else int(age)
        laps.loc[in_stint, "stint"] = int(stint)
        laps.loc[in_stint, "compound"] = record.get("compound")
        laps.loc[in_stint, "tyre_age_at_start"] = age
        # Tyre age is laps already run on that set, counted from the tyre's
        # first ever lap, so it carries a set fitted before the race.
        laps.loc[in_stint, "tyre_age"] = age + (
            laps.loc[in_stint, "lap_number"] - int(lap_start) + 1
        )
    return laps


def _representative_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """One representative lap per driver, used for the telemetry download.

    Per driver, take the lap whose duration is closest to the median of the
    driver's green-flag laps in its longest stint: not the quickest lap (which
    is skewed by a tow, a light fuel load or a clear track) and not an in/out
    lap, so the telemetry traces compare like with like.
    """
    valid = laps[laps["lap_duration"].notna()].copy()
    if valid.empty:
        return valid

    if "is_pit_out_lap" in valid.columns:
        pit_out = valid["is_pit_out_lap"].fillna(False).astype(bool)
        valid = valid[~pit_out]

    if "stint" in valid.columns and valid["stint"].notna().any():
        longest = (
            valid.groupby(["driver_number", "stint"], dropna=True)
            .size()
            .reset_index(name="laps")
            .sort_values(["driver_number", "laps"])
            .drop_duplicates("driver_number", keep="last")
        )
        valid = valid.merge(
            longest[["driver_number", "stint"]], on=["driver_number", "stint"], how="inner"
        )
    if valid.empty:
        return valid

    # Laps much slower than the driver's own median are neutralised or
    # incident-affected and are not representative of the compound's pace.
    medians = valid.groupby("driver_number")["lap_duration"].transform("median")
    valid = valid[valid["lap_duration"] <= medians * 1.03]
    if valid.empty:
        return valid

    chosen = (
        valid.assign(_gap=(valid["lap_duration"] - medians).abs())
        .sort_values(["driver_number", "_gap"])
        .drop_duplicates("driver_number", keep="first")
    )
    return chosen


def _fetch_telemetry(
    session_key: int, laps: pd.DataFrame, out: Path, *, refresh: bool = False
) -> int:
    """Save 4 Hz car data for each driver's representative lap."""
    telemetry_path = out / "telemetry.csv.gz"
    if telemetry_path.exists():
        telemetry_path.unlink()

    chosen = _representative_laps(laps)
    if chosen.empty:
        return 0

    parts: list[pd.DataFrame] = []
    for record in chosen.to_dict("records"):
        driver = int(record["driver_number"])
        lap = int(record["lap_number"])
        rows = _get(
            "car_data",
            session_key=session_key,
            driver_number=driver,
            lap_number=lap,
            refresh=refresh,
        )
        if not rows:
            continue
        frame = _to_frame(rows)
        frame["driver_number"] = driver
        frame["lap_number"] = lap
        frame["compound"] = record.get("compound")
        parts.append(frame)
    if not parts:
        return 0
    combined = pd.concat(parts, ignore_index=True)
    return _write_csv(combined, telemetry_path)


def fetch_session(session_key: int, telemetry: bool, *, refresh: bool = False) -> dict:
    """Download one session and return a summary of what was written."""
    out = ROOT / "data" / "sessions" / str(session_key)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, int] = {}

    meta_rows = _sessions_detail(session_key)
    meta = meta_rows[0] if meta_rows else {}
    written["session"] = _write_csv(_to_frame(meta_rows), out / "session.csv")

    # The meeting row is the only place the Grand Prix name exists, so it is
    # saved next to the session and the loader can build a readable label.
    meeting_key = meta.get("meeting_key")
    meeting_rows = (
        _get("meetings", meeting_key=meeting_key) if meeting_key else []
    )
    if meeting_rows:
        written["meeting"] = _write_csv(
            _to_frame(meeting_rows), out / "meeting.csv"
        )

    laps = _to_frame(_get("laps", session_key=session_key, refresh=refresh))
    stints = _to_frame(_get("stints", session_key=session_key, refresh=refresh))
    drivers = _to_frame(_get("drivers", session_key=session_key, refresh=refresh))

    written["drivers"] = _write_csv(drivers, out / "drivers.csv")
    written["stints"] = _write_csv(stints, out / "stints.csv")

    if not laps.empty:
        laps = _attach_stints(laps, stints)
        written["laps"] = _write_csv(laps, out / "laps.csv")
    else:
        written["laps"] = 0

    for name, endpoint in (
        ("race_control", "race_control"),
        ("weather", "weather"),
        ("pit", "pit"),
        ("position", "position"),
    ):
        rows = _get(endpoint, session_key=session_key, refresh=refresh)
        written[name] = _write_csv(_to_frame(rows), out / f"{name}.csv") if rows else 0

    if telemetry and not laps.empty and "lap_duration" in laps.columns:
        written["telemetry"] = _fetch_telemetry(
            session_key, laps, out, refresh=refresh
        )

    return {"session_key": session_key, "meta": meta, "files": written}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, nargs="*", default=SESSION_KEYS)
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="also download 4 Hz car_data (many sessions have none)",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached API responses"
    )
    args = parser.parse_args(argv)

    for key in args.sessions:
        try:
            summary = fetch_session(
                key, telemetry=args.telemetry, refresh=args.refresh
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"session {key}: FAILED ({exc})", file=sys.stderr)
            continue
        meta = summary["meta"]
        name = f"{meta.get('year', '?')} {meta.get('location', '?')} {meta.get('session_name', '?')}"
        files = ", ".join(f"{k}={v}" for k, v in summary["files"].items())
        print(f"session {key} [{name}]: {files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
