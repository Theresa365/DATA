"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from scada_faults.curation import run_draft_stage2_labels
from scada_faults.dataset import run_prepare_data
from scada_faults.events import run_build_events
from scada_faults.modeling import run_stage1_training, run_stage2_training
from scada_faults.reports import run_report_results


def _parse_root(argv: list[str] | None = None) -> Path | None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to the repository root.")
    args = parser.parse_args(argv)
    return args.root


def _emit_paths(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        print(f"{name}: {path}")


def _run_prepare_data(root: Path | None) -> None:
    _emit_paths(run_prepare_data(root))


def _run_build_events(root: Path | None) -> None:
    _emit_paths(run_build_events(root))


def _run_draft_stage2_labels(root: Path | None) -> None:
    print(run_draft_stage2_labels(root))


def _run_train_stage1(root: Path | None) -> None:
    _emit_paths(run_stage1_training(root))


def _run_train_stage2(root: Path | None) -> None:
    _emit_paths(run_stage2_training(root))


def _run_report_results(root: Path | None) -> None:
    _emit_paths(run_report_results(root))


COMMAND_MAP = {
    "prepare-data": _run_prepare_data,
    "build-events": _run_build_events,
    "draft-stage2-labels": _run_draft_stage2_labels,
    "train-stage1": _run_train_stage1,
    "train-stage2": _run_train_stage2,
    "report-results": _run_report_results,
}


def main_prepare_data() -> None:
    _run_prepare_data(_parse_root())


def main_build_events() -> None:
    _run_build_events(_parse_root())


def main_draft_stage2_labels() -> None:
    _run_draft_stage2_labels(_parse_root())


def main_train_stage1() -> None:
    _run_train_stage1(_parse_root())


def main_train_stage2() -> None:
    _run_train_stage2(_parse_root())


def main_report_results() -> None:
    _run_report_results(_parse_root())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SCADA fault-classification workflow CLI.")
    parser.add_argument("command", choices=sorted(COMMAND_MAP))
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to the repository root.")
    args = parser.parse_args(argv)
    COMMAND_MAP[args.command](args.root)


if __name__ == "__main__":
    main()
