# f1-tyre-dashboard — tyre performance and degradation

**Level 2 of the Formula 1 data portfolio.** Take *one* Grand Prix and explain
its tyre story: how each compound degrades, when it falls off a cliff, what a
pit stop actually cost, and which stint plan those numbers favour.

Everything is measured from real OpenF1 timing data rather than asserted. The
three races that ship in [data/sessions](data/sessions) are the ones the reports
below were generated from.

| Session | Key | Race |
|---|---|---|
| 9912 | 2025 | Italian Grand Prix, Monza |
| 11334 | 2026 | Belgian Grand Prix, Spa-Francorchamps |
| 11361 | 2026 | Italian Grand Prix, Monza |

## What it measures

| Question | Method | Output |
|---|---|---|
| Per-compound pace and degradation | One OLS line per stint, then pooled per compound with stint length as the weight | `time = intercept + slope x tyre_age`, plus a slope standard error |
| Cliff age | Laps binned by tyre age; the first bin sitting more than 0.75s above the fitted line | the age a strategist stops planning around |
| Pit-lane time loss | In-lap and out-lap measured against each driver's own median pace either side of the stop | median loss, spread, and a confidence grade |
| Strategy outcome | The fitted curves replayed over the race distance, with the pit loss charged to the stop lap | ranked plans, fastest lap and last tyre age |

Why pool the slopes with stint length as the weight: a five-lap stint and a
thirty-lap stint should not vote equally on how a compound degrades. Why take
the intercept from the *median* of its stints' intercepts: one quick car on a
light fuel load must not set the field's base pace.

## Usage

```bash
pip install -e ".[dev]"

python -m f1tyre sessions                       # the races on disk
python -m f1tyre stints --session 11361         # per-stint and per-compound fits
python -m f1tyre strategy --min-stint 8         # ranked stint plans
python -m f1tyre strategy --safety-car          # same, with a Safety Car on lap 25
python -m f1tyre report                         # write reports/RESULTS.md
python -m f1tyre report --session 11361 --path report.md
python -m f1tyre dashboard --html               # write reports/tyre_dashboard_<slug>.html
```

Also installed as the `f1tyre` console script. Every command defaults to the
newest race and accepts `--session` with either a session key or a directory
slug — except `report`, which defaults to every session in
[data/sessions](data/sessions) and takes `--session` to narrow it to one race.
`--path` sets the output location and `--top` how many strategies to list per
race.

## Data

Fetched from the [OpenF1](https://openf1.org) REST API by
[scripts/fetch_openf1.py](scripts/fetch_openf1.py):

```bash
python scripts/fetch_openf1.py          # fetch a session, or re-use the CSV cache
python scripts/fetch_openf1.py --list   # the keys already downloaded
python scripts/fetch_openf1.py --refresh
```

Responses are cached on disk under `data/cache/openf1/`, and the per-session CSVs
under `data/sessions/<key>/` are what the rest of the code reads. The lap table
is normalised on load: stint and compound are joined in, `lap_seconds` is parsed
from the OpenF1 duration strings, and Safety Car and red-flag windows are read
from the `race_control` feed rather than guessed from slow lap times.

## Tests

```bash
pytest -q        # session loading, degradation fits, pit loss and plan simulation
ruff check .
```

The suite runs against the committed CSVs, so it needs no network and no
fixtures invented by hand — if the fetch script ever produced a table missing
`tyre_age`, or a stint that does not restart at tyre age 1, the tests fail.

## HTML dashboard

`python -m f1tyre dashboard` renders an interactive Plotly report per race:
degradation curves with a fitted confidence band, a stint timeline for the whole
field, and the ranked strategy table. Reports are written under `reports/`, which
is gitignored because they are generated output.

## Markdown report

`python -m f1tyre report` writes the same numbers as one Markdown file — the
repository's own [reports/RESULTS.md](reports/RESULTS.md) is generated this way,
covering all three committed races. Like the dashboard, it lives under the
gitignored `reports/` directory, so pass `--path` to keep a copy anywhere in the
tree.

## License

MIT — see [LICENSE](LICENSE).
