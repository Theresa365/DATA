"""Filesystem helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    outputs: Path
    prepared: Path
    events: Path
    annotations: Path
    stage1: Path
    stage2: Path
    reports: Path
    figures: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_paths(root: Path | None = None) -> ProjectPaths:
    base = root or repo_root()
    outputs = base / "outputs"
    return ProjectPaths(
        root=base,
        outputs=outputs,
        prepared=outputs / "prepared",
        events=outputs / "events",
        annotations=outputs / "annotations",
        stage1=outputs / "stage1",
        stage2=outputs / "stage2",
        reports=outputs / "reports",
        figures=outputs / "figures",
    )


def ensure_output_dirs(root: Path | None = None) -> ProjectPaths:
    paths = project_paths(root)
    for path in [
        paths.outputs,
        paths.prepared,
        paths.events,
        paths.annotations,
        paths.stage1,
        paths.stage2,
        paths.reports,
        paths.figures,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def discover_raw_workbooks(root: Path | None = None) -> dict[str, Path]:
    base = root or repo_root()
    mapping = {
        "distribution": list(base.glob("*Distribution only*.xlsx")),
        "system": list(base.glob("*system type*.xlsx")),
    }
    resolved: dict[str, Path] = {}
    for source_name, matches in mapping.items():
        if not matches:
            raise FileNotFoundError(f"Could not locate workbook for source '{source_name}' in {base}")
        resolved[source_name] = matches[0]
    return resolved
