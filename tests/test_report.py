"""The ``f1tyre report`` subcommand, driven through ``main()``.

These tests run the command the way a reader does rather than calling the
Markdown renderers directly, because the subcommand once failed on *every*
invocation: its ``--out`` flag wrote its path onto ``args.out``, the stdout
stream ``main()`` installs for every subcommand, so the first status write
raised ``AttributeError: 'str' object has no attribute 'write'``. Only a test
that shells out to the command can keep that from coming back.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from f1tyre import cli
from f1tyre.session import SessionCatalogue

#: the ``src`` directory holding the package, for the subprocess checks below
SRC_DIR = Path(cli.__file__).resolve().parents[1]

# 2026 Italian Grand Prix: the default session, and the one the committed
# report was regenerated from.
MONZA_2026 = 11361

HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
CURVE_HEADER = (
    "| tyre | stints | laps | base lap | s/lap | +/- | R2 | cliff | oldest |"
)


# --------------------------------------------------------------- fixture helpers
def _run_report(path, *extra: str) -> int:
    return cli.main(["report", "--path", str(path), *extra])


def _section(text: str, label: str) -> str:
    """The block of ``text`` between ``## label`` and the next ``##`` heading."""

    match = re.search(rf"^## {re.escape(label)}$.*?(?=^## |\Z)", text,
                      re.MULTILINE | re.DOTALL)
    assert match, f"no '## {label}' section in the report"
    return match.group(0)


def _body_rows(section: str) -> list[str]:
    """The data rows of every Markdown table in ``section``."""

    return [
        line
        for line in section.splitlines()
        if line.startswith("| ") and not set(line) <= set("|- ")
    ]


def _table_with_header(section: str, header: str) -> list[str]:
    """Body rows of the table in ``section`` that opens with ``header``."""

    lines = section.splitlines()
    index = next(
        (i for i, line in enumerate(lines) if line == header),
        None,
    )
    assert index is not None, f"no table with header {header!r}:\n{section}"
    rows = []
    for line in lines[index + 1 :]:
        if not line.startswith("| "):
            break
        if set(line) <= set("|- "):
            continue
        rows.append(line)
    assert rows, f"table {header!r} has no data rows"
    return rows


# ------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def all_sessions_report(tmp_path_factory) -> str:
    """The committed invocation: every downloaded race, default flags."""

    path = tmp_path_factory.mktemp("all") / "RESULTS.md"
    assert _run_report(path) == 0
    assert path.exists()
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def monza_report(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("monza") / "RESULTS.md"
    assert _run_report(path, "--session", str(MONZA_2026)) == 0
    return path.read_text(encoding="utf-8")


# ------------------------------------------------------- the file is written
def test_report_writes_a_non_empty_markdown_file(tmp_path):
    path = tmp_path / "RESULTS.md"
    assert _run_report(path, "--session", str(MONZA_2026)) == 0
    assert path.exists(), "the report file was not created"
    assert path.stat().st_size > 0, "the report file is empty"
    text = path.read_text(encoding="utf-8")
    assert text.strip()
    # a title, a pit-loss section and a method note, at minimum
    assert text.startswith("# ")
    for heading in ("## Pit loss, measured per race", "## Method"):
        assert heading in text


def test_report_overwrites_a_stale_file(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text("stale content from an earlier run\n", encoding="utf-8")
    assert _run_report(path, "--session", str(MONZA_2026)) == 0
    assert "stale content" not in path.read_text(encoding="utf-8")


def test_report_reports_the_path_it_wrote(tmp_path, capsys):
    path = tmp_path / "out.md"
    assert _run_report(path, "--session", str(MONZA_2026)) == 0
    # the message goes to the stream main() installs, not a re-opened stdout
    assert f"wrote {path}" in capsys.readouterr().out


def test_report_defaults_to_the_artifacts_directory(tmp_path, monkeypatch):
    # ARTIFACTS_DIR normally points at the committed reports/; redirect it so
    # the test cannot touch the file this repo ships
    monkeypatch.setattr(cli, "ARTIFACTS_DIR", tmp_path)
    assert cli.main(["report", "--session", str(MONZA_2026)]) == 0
    assert (tmp_path / "RESULTS.md").exists()


# --------------------------------------------------------------- a table per session
def test_report_has_a_section_for_every_downloaded_session(
    all_sessions_report: str, sessions
):
    # the default invocation covers the catalogue, not just the newest race
    headings = HEADING.findall(all_sessions_report)
    for race in sessions:
        assert race.label in headings, f"{race.label} is missing from the report"
    assert len(headings) >= len(sessions)


def test_report_session_table_lists_every_session(
    all_sessions_report: str, sessions
):
    for race in sessions:
        # the session summary table is keyed on session number and label
        assert f"| {race.session_key} | {race.label} |" in all_sessions_report
        assert f"| {race.total_laps} |" in all_sessions_report


def test_report_each_session_carries_its_own_curve_table(
    all_sessions_report: str, sessions
):
    for race in sessions:
        section = _section(all_sessions_report, race.label)
        assert "### Pooled compound curves" in section
        rows = _table_with_header(section, CURVE_HEADER)
        # one row per compound this race actually ran, and only those
        compounds = {
            row.split("|")[1].strip() for row in rows
        }
        assert compounds
        assert compounds <= set(race.compounds)
        assert len(rows) == len(compounds), f"duplicate rows for {race.label}"


def test_report_monza_numbers_match_the_committed_data(monza_report: str):
    section = _section(monza_report, "2026 Italian Grand Prix")
    assert "Monza, 53 laps, reference lap 86.752, 28 stints fitted." in section
    assert "### Strategy estimates" in section
    rows = _table_with_header(
        section,
        "| # | strategy | race time | gap | stops | pit time | final tyre age |",
    )
    # ranks run 1..n and the race time is a clock, not a float
    for position, row in enumerate(rows, start=1):
        cells = [cell.strip() for cell in row.split("|")[1:-1]]
        assert cells[0] == str(position)
        assert cells[1]
        assert re.fullmatch(r"\d+:\d{2}\.\d{3}", cells[2]), cells[2]


def test_session_flag_narrows_the_report_to_one_race(
    monza_report: str, all_sessions_report: str, sessions
):
    # --session 11361 is Monza, so no other race may appear
    assert "## 2026 Italian Grand Prix" in monza_report
    others = [r.label for r in sessions if r.session_key != MONZA_2026]
    for label in others:
        assert label not in monza_report
        assert label in all_sessions_report
    assert HEADING.findall(monza_report).count("2026 Italian Grand Prix") == 1


def test_report_accepts_a_slug_as_well_as_a_key(
    catalogue: SessionCatalogue, tmp_path
):
    record = next(r for r in catalogue if r["session_key"] == MONZA_2026)
    path = tmp_path / "RESULTS.md"
    assert _run_report(path, "--session", record["slug"]) == 0
    assert f"| {MONZA_2026} | {record['label']} |" in path.read_text(encoding="utf-8")


# ----------------------------------------------------------------- filtering
def test_report_top_limits_the_strategy_rows(monza_report: str, tmp_path):
    wide = _table_with_header(
        _section(monza_report, "2026 Italian Grand Prix"),
        "| # | strategy | race time | gap | stops | pit time | final tyre age |",
    )
    assert len(wide) == 5, "the default --top is five rows per race"

    path = tmp_path / "top.md"
    assert _run_report(path, "--session", str(MONZA_2026), "--top", "3") == 0
    narrow = _table_with_header(
        _section(path.read_text(encoding="utf-8"), "2026 Italian Grand Prix"),
        "| # | strategy | race time | gap | stops | pit time | final tyre age |",
    )
    assert len(narrow) == 3
    # narrowing the list must not reorder it: rank 1 stays rank 1
    assert narrow[0] == wide[0]


def test_report_min_stint_and_min_laps_are_accepted(monza_report: str, tmp_path):
    path = tmp_path / "filtered.md"
    assert (
        _run_report(
            path,
            "--session",
            str(MONZA_2026),
            "--min-stint",
            "6",
            "--min-laps",
            "6",
        )
        == 0
    )
    text = path.read_text(encoding="utf-8")
    assert text.strip()
    # the tuned flags still produce the same section and curve table
    assert "## 2026 Italian Grand Prix" in text
    assert _table_with_header(_section(text, "2026 Italian Grand Prix"), CURVE_HEADER)


# ------------------------------------------------------------- failure paths
def test_report_unknown_session_fails_cleanly():
    with pytest.raises(SystemExit, match="not downloaded"):
        cli.main(["report", "--session", "nope"])
    # the message names the session and lists what *is* available
    try:
        cli.main(["report", "--session", "nope"])
    except SystemExit as exc:
        message = str(exc)
    assert "nope" in message
    assert str(MONZA_2026) in message


def test_report_unknown_session_exits_nonzero_without_a_traceback(tmp_path):
    # the real console entry point, where the interpreter's SystemExit handler
    # does print: a usage error must be one line, not a traceback
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "f1tyre",
            "report",
            "--session",
            "nope",
            "--path",
            str(tmp_path / "RESULTS.md"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
    )
    assert done.returncode != 0
    assert "Traceback" not in done.stderr
    assert "not downloaded" in done.stderr
    # no report is left half-written, and nothing is claimed on stdout
    assert not (tmp_path / "RESULTS.md").exists()
    assert not done.stdout


def test_report_does_not_clobber_the_stdout_handle(tmp_path, capsys):
    """The original bug: the output flag used to overwrite ``args.out``."""

    before = sys.stdout
    assert _run_report(tmp_path / "RESULTS.md", "--session", str(MONZA_2026)) == 0
    assert sys.stdout is before
    assert capsys.readouterr().out.strip()
