"""EDA figures and Markdown reporting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scada_faults.curation import load_stage2_annotations
from scada_faults.dataset import run_prepare_data
from scada_faults.events import load_events, run_build_events
from scada_faults.modeling import run_stage1_training, run_stage2_training
from scada_faults.paths import ensure_output_dirs


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_bar_plot(series: pd.Series, output_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_figures(root: Path | None = None) -> dict[str, Path]:
    paths = ensure_output_dirs(root)
    distribution_events = load_events(root, source_name="distribution")
    stage2_annotations = load_stage2_annotations(root)

    figure_paths = {
        "distribution_events_by_month": paths.figures / "distribution_events_by_month.png",
        "stage1_label_counts": paths.figures / "stage1_label_counts.png",
        "weather_counts": paths.figures / "distribution_weather_counts.png",
        "stage2_label_counts": paths.figures / "stage2_label_counts.png",
    }

    monthly_counts = (
        distribution_events.loc[~distribution_events["is_chrono_anomaly"].fillna(False)]
        .assign(event_month_key=lambda df: df["event_date"].dt.to_period("M").astype(str))
        .groupby("event_month_key")
        .size()
    )
    stage1_counts = distribution_events["stage1_binary_label"].value_counts()
    weather_counts = distribution_events["weather"].value_counts()
    stage2_counts = stage2_annotations["final_label"].value_counts()

    _save_bar_plot(monthly_counts, figure_paths["distribution_events_by_month"], "Distribution Events By Month", "Events")
    _save_bar_plot(stage1_counts, figure_paths["stage1_label_counts"], "Stage 1 Label Counts", "Events")
    _save_bar_plot(weather_counts, figure_paths["weather_counts"], "Distribution Weather Counts", "Events")
    _save_bar_plot(stage2_counts, figure_paths["stage2_label_counts"], "Stage 2 Final Label Counts", "Events")
    return figure_paths


def build_markdown_report(root: Path | None = None) -> str:
    paths = ensure_output_dirs(root)
    distribution_events = load_events(root, source_name="distribution")
    system_events = load_events(root, source_name="system")
    annotations = load_stage2_annotations(root)

    stage1_metrics = _load_json(paths.stage1 / "stage1_metrics.json")
    stage2_metrics = _load_json(paths.stage2 / "stage2_metrics.json")
    distribution_rows = _load_json(paths.prepared / "distribution_summary.json")
    system_rows = _load_json(paths.prepared / "system_summary.json")

    clean_distribution_events = distribution_events.loc[~distribution_events["is_chrono_anomaly"].fillna(False)]
    unknown_labels = int((annotations["final_label"] == "unknown/unclassifiable").sum())

    lines = [
        "# SCADA Fault Classification Report",
        "",
        "## Data Summary",
        f"- Distribution workbook rows: {distribution_rows['rows']}",
        f"- System workbook rows: {system_rows['rows']}",
        f"- Distribution event records: {len(distribution_events)}",
        f"- System event records: {len(system_events)}",
        f"- Chronology anomalies quarantined from modeling: {int(distribution_events['is_chrono_anomaly'].sum())}",
        "",
        "## Stage 1 Benchmark",
        "- Target: `Permanent` vs `Non-permanent` at event level.",
        f"- Train/validation events: {stage1_metrics['train_events']}",
        f"- Holdout events: {stage1_metrics['holdout_events']}",
        f"- Selected model: {stage1_metrics['selected_model']}",
        f"- Selected threshold: {stage1_metrics['selected_threshold']:.2f}",
        f"- Holdout macro F1: {stage1_metrics['comparisons'][stage1_metrics['selected_model']]['macro_f1']:.3f}",
        f"- Holdout weighted F1: {stage1_metrics['comparisons'][stage1_metrics['selected_model']]['weighted_f1']:.3f}",
        "",
        "## Stage 2 Fault-Family Study",
        "- Taxonomy: `ground-related`, `phase-to-phase`, `three-phase`, `transformer/internal`, `operational-other`, `unknown/unclassifiable`.",
        f"- Curated distribution events: {len(annotations)}",
        f"- Unknown/unclassifiable events retained for QA only: {unknown_labels}",
        f"- Train/validation events: {stage2_metrics['train_events']}",
        f"- Holdout events: {stage2_metrics['holdout_events']}",
        f"- Selected model: {stage2_metrics['selected_model']}",
        f"- Holdout macro F1: {stage2_metrics['comparisons'][stage2_metrics['selected_model']]['macro_f1']:.3f}",
        f"- Holdout weighted F1: {stage2_metrics['comparisons'][stage2_metrics['selected_model']]['weighted_f1']:.3f}",
    ]

    class_merge_mapping = stage2_metrics.get("class_merge_mapping", {})
    if class_merge_mapping:
        lines.extend(["", "## Stage 2 Rare-Class Merge", f"- Applied mapping: `{class_merge_mapping}`"])

    lines.extend(
        [
            "",
            "## Notes",
            f"- Clean distribution modeling window spans {clean_distribution_events['event_date'].min().date()} to {clean_distribution_events['event_date'].max().date()}.",
            "- Results should be treated as exploratory because the event sample is small and utility-specific.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report_results(root: Path | None = None) -> dict[str, Path]:
    paths = ensure_output_dirs(root)
    if not (paths.prepared / "distribution_rows.csv").exists():
        run_prepare_data(root)
    if not (paths.events / "distribution_events.csv").exists():
        run_build_events(root)
    if not (paths.annotations / "distribution_stage2_annotations.csv").exists():
        load_stage2_annotations(root)
    if not (paths.stage1 / "stage1_metrics.json").exists():
        run_stage1_training(root)
    if not (paths.stage2 / "stage2_metrics.json").exists():
        run_stage2_training(root)

    figure_paths = generate_figures(root)
    report_markdown = build_markdown_report(root)
    report_path = paths.reports / "summary.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    output_paths = {"report": report_path}
    output_paths.update(figure_paths)
    return output_paths
