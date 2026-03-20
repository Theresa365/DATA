# SCADA Fault Classification

This repository turns the Botswana Power Corporation SCADA fault workbooks into a small reproducible machine-learning project built around event-level fault incidents.

## Workflow

Install the project in editable mode:

```powershell
python -m pip install -e .[dev]
```

Run the pipeline:

```powershell
python -m scada_faults.cli prepare-data
python -m scada_faults.cli build-events
python -m scada_faults.cli draft-stage2-labels
python -m scada_faults.cli train-stage1
python -m scada_faults.cli train-stage2
python -m scada_faults.cli report-results
```

Generated artifacts are written under `outputs/`.

## What The Project Does

- Reads the two Excel workbooks in the repo root.
- Normalizes mixed date and time formats and quarantines date anomalies.
- Collapses apparatus rows into one event-level sample per fault incident.
- Benchmarks a stage-1 operational model for `Permanent` vs `Non-permanent`.
- Generates and curates stage-2 electrical fault-family labels for distribution events.
- Produces metrics, predictions, figures, and a Markdown summary report.

## Project Layout

- `src/scada_faults/`: package code and CLI entry points
- `tests/`: parser, event-building, leakage, and smoke tests
- `notebooks/`: exploratory notebook scaffold
- `outputs/`: generated data, metrics, and reports
