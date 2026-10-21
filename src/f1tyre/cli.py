"""Command line interface to the tyre analysis.

Five subcommands, all reading the same offline CSVs the fetch script wrote:

- ``sessions``   what has been downloaded
- ``stints``     per-stint degradation fits, pooled compound curves, pit loss
- ``strategy``   replay candidate stint plans through those curves
- ``report``     the same numbers as one Markdown file under reports/
- ``dashboard``  a self-contained HTML report of all of the above

Every number the CLI prints comes from ``RaceSession``/``strategy`` - this
module only formats and selects, so a claim in the output can be traced back
to the function that made it.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

from .session import SessionCatalogue, load_session
from .strategy import (
    COMPOUND_ORDER,
    MIN_LAPS_FOR_FIT,
    StrategyPlan,
    degradation_curves,
    measured_pit_loss,
    stint_degradation,
    strategy_comparison,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "reports"


# ------------------------------------------------------------------ formatting
def _table(
    rows: list[dict], columns: list[tuple[str, str]], empty: str = "  (none)"
) -> str:
    """Render rows as an aligned text table; ``columns`` are (key, header)."""

    if not rows:
        return empty + "\n"
    widths = {
        key: max(len(header), *(len(str(row.get(key, ""))) for row in rows))
        for key, header in columns
    }
    rule = "  ".join("-" * widths[key] for key, _ in columns)
    head = "  ".join(header.ljust(widths[key]) for key, header in columns)
    body = "\n".join(
        "  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns)
        for row in rows
    )
    return f"{head}\n{rule}\n{body}\n"


def _fmt_clock(seconds: float) -> str:
    total = max(0.0, float(seconds))
    return f"{int(total // 60)}:{total % 60:06.3f}"


def _fmt_laptime(seconds: float) -> str:
    return f"{float(seconds):.3f}"


# -------------------------------------------------------------------- helpers
def _catalogue() -> SessionCatalogue:
    catalogue = SessionCatalogue()
    if not len(catalogue):
        raise SystemExit(
            "no downloaded sessions found - run scripts/fetch_openf1.py first"
        )
    return catalogue


def _pick(catalogue: SessionCatalogue, requested: str | None):
    """Resolve a session argument: an int key, a slug, or the newest race."""

    records = list(catalogue)
    if requested is None:
        default = catalogue.default()
        return next(r for r in records if r["session_key"] == default)
    wanted = str(requested)
    for record in records:
        if str(record["session_key"]) == wanted or record["slug"] == wanted:
            return record
    raise SystemExit(
        f"session {requested!r} not downloaded (have: "
        + ", ".join(str(r["session_key"]) for r in records)
        + ")"
    )


def _safety_car_window(session) -> tuple[int, int] | None:
    """The first contiguous block of neutralised laps, for the SC discount."""

    laps = session.laps
    flagged = sorted(
        int(v)
        for v in laps.loc[laps["neutralised"].astype(bool), "lap_number"].unique()
    )
    if not flagged:
        return None
    return flagged[0], flagged[-1]


def _plans(
    text: list[str] | None, total_laps: int, min_stint: int
) -> list[StrategyPlan]:
    """Parse ``SOFT,MEDIUM:15,31`` plans, or build a sensible default set."""

    if not text:
        return _default_plans(total_laps, min_stint)
    plans: list[StrategyPlan] = []
    for entry in text:
        head, _, stops = entry.partition(":")
        compounds = [c.strip().upper() for c in head.split(",") if c.strip()]
        laps = [int(s) for s in stops.split(",") if s.strip()]
        if len(compounds) != len(laps) + 1:
            raise SystemExit(
                f"plan {entry!r} needs one more compound than stop lap "
                f"(got {len(compounds)} compounds, {len(laps)} stops)"
            )
        unknown = set(compounds) - set(COMPOUND_ORDER)
        if unknown:
            raise SystemExit(f"plan {entry!r} has unknown compounds: {sorted(unknown)}")
        label = " > ".join(compounds)
        if laps:
            label += " (stops " + ", ".join(str(s) for s in laps) + ")"
        plans.append(StrategyPlan(name=label, compounds=compounds, stops=laps))
    return plans


def _default_plans(total_laps: int, min_stint: int) -> list[StrategyPlan]:
    """One-stop for every compound pair, plus a no-stop baseline."""

    names = ["SOFT", "MEDIUM", "HARD"]
    plans: list[StrategyPlan] = []
    for first in names:
        for second in names:
            if first == second:
                continue
            for stop in range(min_stint, total_laps - min_stint + 1):
                plans.append(
                    StrategyPlan(
                        name=f"{first} > {second} on lap {stop}",
                        compounds=[first, second],
                        stops=[stop],
                    )
                )
    # running the race on one set of tyres is a valid baseline, not a strategy
    plans.append(StrategyPlan(name="no stop", compounds=["MEDIUM"], stops=[]))
    return plans


# ------------------------------------------------------------------- commands
def _cmd_sessions(args: argparse.Namespace) -> int:
    rows = [
        {
            "key": record["session_key"],
            "year": record["year"],
            "label": record["label"],
            "circuit": record["circuit"],
            "laps": record["total_laps"],
            "slug": record["slug"],
        }
        for record in _catalogue()
    ]
    args.out.write(
        _table(
            rows,
            [
                ("key", "session"),
                ("year", "year"),
                ("label", "grand prix"),
                ("circuit", "circuit"),
                ("laps", "laps"),
                ("slug", "slug"),
            ],
        )
    )
    return 0


def _cmd_stints(args: argparse.Namespace) -> int:
    catalogue = _catalogue()
    record = _pick(catalogue, args.session)
    session = load_session(record["session_key"], catalogue)
    fits = stint_degradation(session, min_laps=args.min_laps)
    if fits.empty:
        raise SystemExit(f"{session.label}: no stint long enough to fit")

    args.out.write(
        f"{session.label}  ({session.circuit}, {session.total_laps} laps)  "
        f"reference lap {_fmt_laptime(session.reference_time)}\n\n"
    )
    args.out.write(
        _table(
            [
                {
                    "driver": row.code,
                    "stint": int(row.stint_number),
                    "compound": row.compound,
                    "laps": f"{int(row.first_lap)}-{int(row.last_lap)}",
                    "n": int(row.laps),
                    "base": _fmt_laptime(row.base_time),
                    "slope": f"{row.slope:+.4f}",
                    "r2": f"{row.r_squared:.2f}",
                }
                for row in fits.itertuples()
            ],
            [
                ("driver", "driver"),
                ("stint", "stint"),
                ("compound", "tyre"),
                ("laps", "laps"),
                ("n", "n"),
                ("base", "base lap"),
                ("slope", "s/lap"),
                ("r2", "R2"),
            ],
        )
        + "\n"
    )
    curves = degradation_curves(session, fits=fits)
    args.out.write(
        _table(
            [
                {
                    "compound": curve.compound,
                    "stints": curve.stints,
                    "laps": curve.laps,
                    "base": _fmt_laptime(curve.intercept),
                    "slope": f"{curve.slope:+.4f}",
                    "se": f"{curve.slope_se:.4f}",
                    "r2": f"{curve.r_squared:.2f}",
                    "cliff": curve.cliff_age,
                    "oldest": curve.max_age_seen,
                }
                for curve in sorted(
                    curves, key=lambda c: COMPOUND_ORDER.get(c.compound, 9)
                )
            ],
            [
                ("compound", "tyre"),
                ("stints", "stints"),
                ("laps", "laps"),
                ("base", "base lap"),
                ("slope", "s/lap"),
                ("se", "+/-"),
                ("r2", "R2"),
                ("cliff", "cliff"),
                ("oldest", "oldest"),
            ],
        )
        + "\n"
    )
    loss = measured_pit_loss(session)
    args.out.write(
        f"pit loss: {loss.seconds:.2f} s over {loss.stops} stops "
        f"(90-10 spread {loss.spread:.2f} s, confidence {loss.confidence})\n"
    )
    return 0


def _cmd_strategy(args: argparse.Namespace) -> int:
    catalogue = _catalogue()
    record = _pick(catalogue, args.session)
    session = load_session(record["session_key"], catalogue)
    plans = _plans(args.plan, session.total_laps, args.min_stint)
    window = _safety_car_window(session) if args.safety_car else None
    results = strategy_comparison(
        session,
        plans,
        total_laps=session.total_laps,
        safety_car_laps=window,
    )
    if not results:
        raise SystemExit("no plan could be evaluated on this race")

    best = results[0].total_time
    header = f"{session.label}  {session.total_laps} laps"
    if window:
        header += f"  SC window {window[0]}-{window[1]} (pit loss halved)"
    args.out.write(header + "\n")
    args.out.write(
        _table(
            [
                {
                    "rank": index,
                    "plan": result.name,
                    "time": _fmt_clock(result.total_time),
                    "gap": "-" if index == 1 else f"+{result.total_time - best:.3f} s",
                    "stops": ", ".join(str(s) for s in result.stops) or "-",
                    "last_age": result.last_tyre_age,
                }
                for index, result in enumerate(results, start=1)
            ],
            [
                ("rank", "#"),
                ("plan", "strategy"),
                ("time", "race time"),
                ("gap", "gap"),
                ("stops", "stops"),
                ("last_age", "final tyre age"),
            ],
        )
    )
    return 0


def _degradation_png(session, fits) -> str:
    """Lap time against tyre age, per compound, as an inline PNG data URI."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colours = {"SOFT": "#d62728", "MEDIUM": "#e8a33d", "HARD": "#7f7f7f"}
    figure, axes = plt.subplots(figsize=(7.0, 4.0), dpi=110)
    for compound, group in fits.groupby("compound"):
        colour = colours.get(str(compound), "#1f77b4")
        for row in group.itertuples():
            span = int(row.laps) - 1
            axes.plot(
                [row.mean_tyre_age, row.mean_tyre_age + span],
                [row.base_time, row.base_time + row.slope * span],
                color=colour,
                alpha=0.35,
                linewidth=1.0,
            )
            axes.scatter(
                [row.mean_tyre_age],
                [row.base_time + row.slope * row.mean_tyre_age],
                color=colour,
                s=14,
                alpha=0.8,
            )
        axes.scatter([], [], color=colour, label=str(compound))
    axes.set_xlabel("tyre age (laps)")
    axes.set_ylabel("lap time (s)")
    axes.set_title(f"{session.label} - stint degradation")
    axes.legend(frameon=False)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _cmd_dashboard(args: argparse.Namespace) -> int:
    catalogue = _catalogue()
    record = _pick(catalogue, args.session)
    session = load_session(record["session_key"], catalogue)
    fits = stint_degradation(session)
    if fits.empty:
        raise SystemExit(f"{session.label}: no stint long enough to fit")
    curves = degradation_curves(session, fits=fits)
    loss = measured_pit_loss(session)
    plans = _plans(args.plan, session.total_laps, args.min_stint)
    window = _safety_car_window(session)
    results = strategy_comparison(
        session,
        plans,
        total_laps=session.total_laps,
        safety_car_laps=window,
    )
    best = results[0].total_time if results else 0.0

    chart = _degradation_png(session, fits) if args.charts else ""
    out_path = Path(args.html or ARTIFACTS_DIR / f"tyre_dashboard_{session.slug}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _dashboard_html(session, fits, curves, loss, results, best, chart, window),
        encoding="utf-8",
    )
    args.out.write(f"wrote {out_path}\n")
    return 0


def _report_markdown(analysed: list[dict], *, min_stint: int, top: int) -> str:
    """Every fitted number, as one Markdown file.

    The dashboard is the readable view of a single race; this is the complete
    one: all the races on disk, all their curves, and their measured pit loss
    side by side, so a reader can see which numbers came from the data and
    which were assumed.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fitted = sum(
        int(a["fits"]["laps"].sum()) if not a["fits"].empty else 0 for a in analysed
    )
    parts = [
        _report_header(len(analysed), fitted, stamp),
        _md_table(
            SESSION_COLUMNS,
            [
                {
                    "key": a["session"].session_key,
                    "label": a["session"].label,
                    "circuit": a["session"].circuit,
                    "laps": a["session"].total_laps,
                    "reference": _fmt_laptime(a["session"].reference_time),
                    "fits": len(a["fits"]),
                    "green": int(a["fits"]["laps"].sum()) if not a["fits"].empty else 0,
                }
                for a in analysed
            ],
        ),
        _loss_section(analysed),
    ]
    for entry in analysed:
        parts.append(_report_race(entry, min_stint=min_stint, top=top))
    parts.append(_report_method())
    return "\n".join(p for p in parts if p)


def _report_header(race_count: int, fitted_laps: int, stamp: str) -> str:
    return (
        "# Tyre degradation and strategy estimates\n"
        "\n"
        f"Formula 1, {race_count} Grands Prix, {fitted_laps} laps fitted. "
        f"Generated {stamp} by `f1tyre report` from the CSVs in `data/sessions/`.\n"
        "\n"
        "This report answers three questions with the measured lap data:\n"
        "\n"
        "1. **How fast is each compound on a zero-age tyre, and how much does it\n"
        "   lose per lap of tyre age?**\n"
        "2. **What does a pit stop actually cost in this race, and how certain is\n"
        "   that number?**\n"
        "3. **Which stint plan does that imply for the same race?**\n"
        "\n"
        "Pace and degradation come from ordinary least squares over green-flag laps "
        "only; pit loss is measured against each driver's own pace either side of "
        "the stop. Strategy estimates are those two products replayed over the race "
        "distance - arithmetic over fits, not a prediction of a result.\n"
    )


def _loss_section(analysed: list[dict]) -> str:
    """The pit-loss measurement, which every strategy estimate rests on."""

    rows = [
        {
            "key": a["session"].session_key,
            "loss": f"{a['loss'].seconds:.2f} s" if a["loss"].usable else "not measured",
            "spread": f"{a['loss'].spread:.2f} s",
            "confidence": a["loss"].confidence,
            "stops": a["loss"].stops,
        }
        for a in analysed
    ]
    return "\n".join(
        [
            "## Pit loss, measured per race",
            "",
            _md_table(LOSS_COLUMNS, rows),
            "A stop is timed from the laps actually driven under green flags, "
            "in-lap and out-lap together, against the same driver's own median pace "
            "either side. A confidence grade is printed next to the number because "
            "a spread that wide is not a lap-time prediction: a race whose stops "
            "mostly happened under a Safety Car reports a loss a strategist should "
            "ignore, and the grade is the honest version of that caveat.",
        ]
    )


def _report_race(entry: dict, *, min_stint: int, top: int) -> str:
    """The curves, pit loss and strategy estimates for one race."""

    session = entry["session"]
    fits = entry["fits"]
    curves = entry["curves"]
    loss = entry["loss"]
    lines = [
        f"## {session.label}",
        "",
        f"{session.circuit}, {session.total_laps} laps, reference lap "
        f"{_fmt_laptime(session.reference_time)}, {len(fits)} stints fitted.",
        "",
        "### Pooled compound curves",
        "",
    ]
    if curves:
        lines.append(_md_table(CURVE_COLUMNS, _curve_rows(curves)))
        lines.append(
            "`base lap` is the pace of a zero-age tyre of that compound, taken as "
            "the median of its stints' intercepts so one quick car on a light fuel "
            "load cannot set the field's pace. `s/lap` pools the per-stint slopes "
            "weighted by stint length. R2 is printed because these fits are weak: "
            "a stint is a flat line plus noise, so a low R2 is the measurement "
            "saying most of the lap-time spread is not tyre age, not a fitting "
            "failure."
        )
    else:
        lines.append("_No stint in this race was long enough to fit._")

    if not loss.usable:
        lines.append(
            "\nPit loss was not measured in this race (fewer than four informative "
            "stops), so no strategy estimates are given for it."
        )
        return "\n".join(lines)

    plans = _default_plans(session.total_laps, min_stint)
    try:
        results = strategy_comparison(
            session, plans, fits=fits, total_laps=session.total_laps, pit_loss=loss
        )
    except ValueError as exc:
        lines.append(f"\nStrategy could not be estimated: {exc}")
        return "\n".join(lines)

    best = results[0].total_time
    lines += [
        "",
        "### Strategy estimates",
        "",
        f"Every one-stop plan plus the no-stop baseline, replayed through the curves "
        f"above with {loss.seconds:.2f} s charged to each stop lap and the fitted "
        f"slope doubled past the cliff age. Best race time {_fmt_clock(best)}; "
        f"showing the best {min(top, len(results))} of {len(results)}.",
        "",
    ]
    window = _safety_car_window(session)
    if window:
        lines.append(
            f"This race was neutralised on laps {window[0]}-{window[1]}; those laps "
            "are excluded from the fits above and from the pit-loss measurement.\n"
        )
    lines.append(_md_table(RESULT_COLUMNS, _result_rows(results[:top], best)))
    return "\n".join(lines)


def _report_method() -> str:
    return "\n".join(
        [
            "## Method",
            "",
            "Lap times come from the committed OpenF1 snapshot per session: in/out "
            "laps, lapped traffic and Safety Car laps are excluded before fitting, "
            "and Safety Car windows are read from the `race_control` feed rather "
            "than guessed from slow lap times. Each stint is fitted as "
            "`lap_time = intercept + slope x tyre_age` by ordinary least squares; "
            "compound curves pool those slopes weighted by stint length, and the "
            "cliff age is the first tyre-age bin sitting more than 0.75 s above the "
            "fitted line. Pit loss is the median in-lap plus out-lap cost over "
            "every stop actually made, with a confidence grade from its own spread. "
            "Strategy estimates use nothing from outside the race they are scored "
            "on: the same curves, the same pit loss, the same lap count.",
            "",
        ]
    )


def _cmd_report(args: argparse.Namespace) -> int:
    catalogue = _catalogue()
    records = [_pick(catalogue, args.session)] if args.session else list(catalogue)
    sessions = [load_session(record["session_key"], catalogue) for record in records]
    analysed = _analyse(sessions, args.min_laps)
    text = _report_markdown(analysed, min_stint=args.min_stint, top=args.top)
    out_path = Path(args.path or ARTIFACTS_DIR / "RESULTS.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    args.out.write(f"wrote {out_path}\n")
    return 0


def _escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def _html_table(columns: list[tuple[str, str]], rows: list[dict]) -> str:
    """Table rows as HTML, or a plain note when there is nothing to show."""

    if not rows:
        return '<p class="note">nothing to show</p>'
    head = "".join(f"<th>{_escape(header)}</th>" for _, header in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(row.get(key, ''))}</td>" for key, _ in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


STINT_COLUMNS = [
    ("driver", "driver"),
    ("stint", "stint"),
    ("compound", "tyre"),
    ("laps", "laps"),
    ("n", "n"),
    ("base", "base lap"),
    ("slope", "s/lap"),
    ("r2", "R2"),
]


def _stint_rows(fits) -> list[dict]:
    return [
        {
            "driver": row.code,
            "stint": int(row.stint_number),
            "compound": row.compound,
            "laps": f"{int(row.first_lap)}-{int(row.last_lap)}",
            "n": int(row.laps),
            "base": _fmt_laptime(row.base_time),
            "slope": f"{row.slope:+.4f}",
            "r2": f"{row.r_squared:.2f}",
        }
        for row in fits.itertuples()
    ]


CURVE_COLUMNS = [
    ("compound", "tyre"),
    ("stints", "stints"),
    ("laps", "laps"),
    ("base", "base lap"),
    ("slope", "s/lap"),
    ("se", "+/-"),
    ("r2", "R2"),
    ("cliff", "cliff"),
    ("oldest", "oldest"),
]


def _curve_rows(curves) -> list[dict]:
    return [
        {
            "compound": curve.compound,
            "stints": curve.stints,
            "laps": curve.laps,
            "base": _fmt_laptime(curve.intercept),
            "slope": f"{curve.slope:+.4f}",
            "se": f"{curve.slope_se:.4f}",
            "r2": f"{curve.r_squared:.2f}",
            "cliff": curve.cliff_age,
            "oldest": curve.max_age_seen,
        }
        for curve in sorted(curves, key=lambda c: COMPOUND_ORDER.get(c.compound, 9))
    ]


RESULT_COLUMNS = [
    ("rank", "#"),
    ("plan", "strategy"),
    ("time", "race time"),
    ("gap", "gap"),
    ("stops", "stops"),
    ("pit", "pit time"),
    ("last_age", "final tyre age"),
]


def _result_rows(results, best: float) -> list[dict]:
    return [
        {
            "rank": index,
            "plan": result.name,
            "time": _fmt_clock(result.total_time),
            "gap": "-" if index == 1 else f"+{result.total_time - best:.3f} s",
            "stops": ", ".join(str(s) for s in result.stops) or "-",
            "pit": f"{result.pit_time:.1f} s",
            "last_age": result.last_tyre_age,
        }
        for index, result in enumerate(results, start=1)
    ]




# -------------------------------------------------------------------- report
def _md_table(columns: list[tuple[str, str]], rows: list[dict]) -> str:
    """Rows as a Markdown table; ``columns`` are (key, header).

    Deliberately not shared with ``_html_table``: the committed report has to
    render on GitHub with no build step, so it stays plain text.
    """

    if not rows:
        return "_nothing to show_\n"
    head = "| " + " | ".join(header for _, header in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    )
    return f"{head}\n{rule}\n{body}\n"


def _analyse(sessions: list, min_laps: int) -> list[dict]:
    """Fit every session once: stints, pooled curves and measured pit loss."""

    analysed = []
    for session in sessions:
        fits = stint_degradation(session, min_laps=min_laps)
        analysed.append(
            {
                "session": session,
                "fits": fits,
                "curves": degradation_curves(session, fits=fits)
                if not fits.empty
                else [],
                "loss": measured_pit_loss(session),
            }
        )
    return analysed


SESSION_COLUMNS = [
    ("key", "session"),
    ("label", "grand prix"),
    ("circuit", "circuit"),
    ("laps", "laps"),
    ("reference", "reference lap"),
    ("fits", "stints fitted"),
    ("green", "green laps fitted"),
]


LOSS_COLUMNS = [
    ("key", "session"),
    ("loss", "loss per stop"),
    ("spread", "90-10 spread"),
    ("confidence", "confidence"),
    ("stops", "stops"),
]


def _dashboard_html(session, fits, curves, loss, results, best, chart, window) -> str:
    """The whole report as one standalone file - no CDN, no external assets."""

    sc = ""
    if window:
        sc = (
            "<p class='note'>Safety Car detected on laps "
            f"{window[0]}-{window[1]}: pit loss is halved inside that "
            "window, matching the <code>--safety-car</code> flag.</p>"
        )
    chart_html = (
        f"<figure><img alt='stint degradation scatter' src='{chart}'></figure>"
        if chart
        else "<p class='note'>charts skipped (<code>--no-charts</code>)</p>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(session.label)} - tyre degradation dashboard</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 15px/1.55 ui-sans-serif, system-ui, sans-serif; margin: 0 auto;
       max-width: 62rem; padding: 2rem 1.25rem 4rem; }}
h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; }}
h2 {{ font-size: 1.05rem; text-transform: uppercase; letter-spacing: .06em;
     margin: 2.5rem 0 .75rem; }}
.meta, .note {{ color: #6b7280; font-size: .9rem; margin: .35rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: .4rem .6rem; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
thead th {{ border-bottom: 2px solid #9ca3af; }}
figure {{ margin: 1.5rem 0; }}
img {{ max-width: 100%; height: auto; border-radius: .5rem; }}
code {{ font-size: .85em; }}
</style>
</head>
<body>
<h1>{_escape(session.label)}</h1>
<p class="meta">{_escape(session.circuit)} &middot; {session.total_laps} laps &middot;
reference lap {_fmt_laptime(session.reference_time)} &middot;
{fits.shape[0]} stints fitted</p>

<h2>Per-stint fits</h2>
{_html_table(STINT_COLUMNS, _stint_rows(fits))}

<h2>Pooled compound curves</h2>
{_html_table(CURVE_COLUMNS, _curve_rows(curves))}

<h2>Pit loss</h2>
<p>Measured {loss.seconds:.2f} s per stop over {loss.stops} stops
(90-10 spread {loss.spread:.2f} s, confidence {_escape(loss.confidence)}).
A stop is only measured from laps that were actually driven under green flags,
and is compared against that driver's own pace either side of the stop.</p>

<h2>Strategy estimates</h2>
{sc}
{_html_table(RESULT_COLUMNS, _result_rows(results, best))}
<p class="note">Estimates replay each plan through the curves above and charge
the measured pit loss in full to the stop lap. Past the cliff age the fitted
degradation is multiplied by 2.0.</p>

<h2>Degradation</h2>
{chart_html}

<h2>Method</h2>
<p class="note">Lap times come from the committed OpenF1 snapshot for this
session; in-/out-laps, lapped traffic and Safety Car laps are excluded before
fitting. Each stint is
<code>lap_time = intercept + slope x tyre_age</code> fitted by ordinary least
squares; compound curves pool those fits weighted by stint length. Generated by
<code>f1tyre dashboard</code>.</p>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1tyre",
        description="tyre degradation and strategy estimates from offline OpenF1 data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--session", default=None, help="session key or slug (default: newest race)")

    sessions = subparsers.add_parser("sessions", help="list the downloaded sessions")
    add_common(sessions)

    stints = subparsers.add_parser("stints", help="per-stint and per-compound fits")
    add_common(stints)
    stints.add_argument("--min-laps", type=int, default=MIN_LAPS_FOR_FIT)

    strategy = subparsers.add_parser("strategy", help="compare stint plans")
    add_common(strategy)
    strategy.add_argument("--min-stint", type=int, default=8)
    strategy.add_argument("--safety-car", action="store_true")
    strategy.add_argument(
        "--plan",
        action="append",
        metavar="COMPOUNDS:STOPS",
        help="e.g. MEDIUM,HARD:27 (repeatable; default scans every one-stop)",
    )

    report = subparsers.add_parser(
        "report", help="write the same numbers as one Markdown file"
    )
    add_common(report)
    report.add_argument("--min-laps", type=int, default=MIN_LAPS_FOR_FIT)
    report.add_argument("--min-stint", type=int, default=8)
    report.add_argument(
        "--path",
        default=None,
        help="output path (default reports/RESULTS.md)",
    )
    report.add_argument(
        "--top",
        type=int,
        default=5,
        help="how many strategies to list per race (default 5)",
    )

    dashboard = subparsers.add_parser("dashboard", help="write the HTML report")
    add_common(dashboard)
    dashboard.add_argument("--min-stint", type=int, default=8)
    dashboard.add_argument("--plan", action="append", metavar="COMPOUNDS:STOPS")
    dashboard.add_argument("--html", default=None, help="output path (default reports/tyre_dashboard_<slug>.html)")
    dashboard.add_argument("--no-charts", dest="charts", action="store_false")

    return parser


COMMANDS = {
    "sessions": _cmd_sessions,
    "stints": _cmd_stints,
    "strategy": _cmd_strategy,
    "report": _cmd_report,
    "dashboard": _cmd_dashboard,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out = sys.stdout
    return COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
