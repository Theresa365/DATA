"""EDA figures and Markdown reporting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scada_faults.curation import load_stage2_annotations
from scada_faults.dataset import run_prepare_data
from scada_faults.events import load_events, run_build_events
from scada_faults.modeling import run_stage1_training, run_stage2_training
from scada_faults.paths import ensure_output_dirs
from scada_faults.validation import export_domain_validation_pack


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_confidence_interval(metrics: dict[str, object], metric_name: str) -> str | None:
    confidence_intervals = metrics.get("confidence_intervals")
    if not isinstance(confidence_intervals, dict):
        return None
    metric_interval = confidence_intervals.get(metric_name)
    if not isinstance(metric_interval, dict):
        return None
    lower = metric_interval.get("lower")
    upper = metric_interval.get("upper")
    if lower is None or upper is None:
        return None
    return f"{float(lower):.3f} to {float(upper):.3f}"


def _format_temporal_backtest_mean(metrics: dict[str, object], metric_name: str) -> float | None:
    temporal_backtest = metrics.get("temporal_backtest")
    if not isinstance(temporal_backtest, dict):
        return None
    summary = temporal_backtest.get("summary")
    if not isinstance(summary, dict):
        return None
    metric_summary = summary.get(metric_name)
    if not isinstance(metric_summary, dict):
        return None
    mean_value = metric_summary.get("mean")
    return None if mean_value is None else float(mean_value)


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


def _draw_fault_sheet_replica(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    line_color = "#111111"
    green = "#1b8a5a"
    font = {"fontfamily": "DejaVu Serif", "color": "#1a1a1a"}

    def hline(y: float, lw: float = 1.0, color: str = line_color) -> None:
        ax.plot([0.5, 99.5], [y, y], color=color, lw=lw)

    def vline(x: float, y0: float, y1: float, lw: float = 1.0, color: str = line_color) -> None:
        ax.plot([x, x], [y0, y1], color=color, lw=lw)

    def label(x: float, y: float, text: str, size: float = 8.6, weight: str = "normal", ha: str = "left") -> None:
        ax.text(x, y, text, fontsize=size, fontweight=weight, ha=ha, va="center", **font)

    hline(51.0, lw=1.6, color=green)
    hline(49.2, lw=1.0)
    label(40, 50.1, "BOTSWANA POWER CORPORATION", size=9.2, weight="bold", ha="center")

    label(1.0, 47.6, "FAULT NO:", size=8.8, weight="bold")
    ax.add_patch(Rectangle((9.5, 46.4), 27, 2.4, fill=False, ec=line_color, lw=1.6))
    label(23.0, 47.6, "060/22", size=8.6, weight="bold", ha="center")

    hline(43.9, lw=1.0)
    for y in [41.5, 39.1, 36.7, 34.3, 31.9, 29.5, 27.1, 24.7, 22.3, 19.9, 17.5, 15.1, 12.7, 10.3, 7.9, 5.5, 3.1]:
        hline(y, lw=0.85)

    for x in [9.5, 36.5, 66.0, 78.0, 85.0, 91.0]:
        vline(x, 21.0, 43.9, lw=1.0)
    vline(99.5, 1.1, 43.9, lw=1.6)
    vline(9.5, 21.0, 48.8, lw=1.0)
    vline(36.5, 21.0, 43.9, lw=1.0)
    vline(66.0, 21.0, 43.9, lw=1.0)
    vline(78.0, 21.0, 43.9, lw=1.0)
    vline(85.0, 39.1, 43.9, lw=1.0)
    vline(91.0, 21.0, 43.9, lw=1.0)

    label(1.0, 42.3, "DAY/DATE", weight="bold")
    label(10.0, 42.3, "Monday 25 February 2024", weight="bold")
    label(36.8, 42.3, "WEATHER", weight="bold")
    label(43.5, 42.3, "CLEAR", weight="bold")
    label(66.3, 42.3, "VOLTAGE LEVEL", size=8.1, weight="bold")
    label(78.4, 42.3, "132/11k", size=8.1, weight="bold")
    label(85.3, 42.3, "AREA:", size=8.1, weight="bold")
    label(95.0, 42.3, "South", size=8.1, weight="bold", ha="center")

    label(1.0, 40.0, "TIME", weight="bold")
    label(10.0, 40.0, "APPARATUS TRIPPED", weight="bold")
    label(36.8, 40.0, "PROTECTION INDICATIONS", weight="bold")
    label(66.3, 40.0, "TIME RECLOSED", size=8.1, weight="bold")
    label(84.5, 40.0, "REPORTED BY", size=8.1, weight="bold", ha="center")
    label(91.3, 40.0, "BREAKER", size=8.1, weight="bold")
    label(91.3, 38.5, "SEQUENCE", size=8.1, weight="bold")

    row_y = [35.5, 33.1, 30.7, 28.3]
    label(10.0, row_y[0], "Gab East 132/11 kV T1B : HV CB 110B")
    label(36.8, row_y[0], "did not trip")

    label(23.7, row_y[1], ": LV CB 1H0B")
    label(36.8, row_y[1], "did not trip")

    label(1.0, row_y[2], "12h52")
    label(10.0, row_y[2], "Gab East 132/11 kV T1A : HV CB 110A")
    label(36.8, row_y[2], "O/C & E/F : A & B To Ground")
    label(66.3, row_y[2], "14h08")
    label(78.5, row_y[2], "SCADA/R. Galetshets")
    label(91.3, row_y[2], "Tripped")

    label(23.7, row_y[3], ": LV CB 1H0A")
    label(36.8, row_y[3], "O/C & E/F : A & B To Ground")
    label(66.3, row_y[3], "14h14")
    label(78.5, row_y[3], "SCADA/R. Galetshets")
    label(91.3, row_y[3], "Tripped")

    hline(21.0, lw=1.2)
    label(1.0, 19.4, "OPERATIONS CARRIED OUT", size=8.8, weight="bold")
    hline(18.4, lw=1.0)
    hline(16.2, lw=1.4, color=green)
    label(
        1.0,
        13.9,
        "CSS requested to close one feeder at a time and 132/11 kV Transformer 1A closed first and Transformer 2A followed and they both held.",
        size=8.4,
    )
    hline(8.3, lw=1.0)
    label(1.0, 4.6, "LINE INSPECTION ETC", size=8.8, weight="bold")
    hline(3.7, lw=1.0)
    label(
        1.0,
        2.1,
        "CSS advised fault was due to one of the feeders CB failed to trip during fault, hence through fault.",
        size=8.4,
    )
    ax.add_patch(Rectangle((0.6, 1.1), 98.9, 15.1, fill=False, ec="none", lw=0))
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
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
        "fault_sheet_replica": paths.figures / "fault_sheet_replica.png",
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
    _draw_fault_sheet_replica(figure_paths["fault_sheet_replica"])
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
    stage1_selected_metrics = stage1_metrics["comparisons"][stage1_metrics["selected_model"]]
    stage2_selected_metrics = stage2_metrics["comparisons"][stage2_metrics["selected_model"]]
    stage1_ci = _format_confidence_interval(stage1_selected_metrics, "macro_f1")
    stage2_ci = _format_confidence_interval(stage2_selected_metrics, "macro_f1")
    stage1_backtest_mean = _format_temporal_backtest_mean(stage1_metrics, "macro_f1")
    stage2_backtest_mean = _format_temporal_backtest_mean(stage2_metrics, "macro_f1")

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
        f"- Holdout macro F1: {stage1_selected_metrics['macro_f1']:.3f}",
        f"- Holdout weighted F1: {stage1_selected_metrics['weighted_f1']:.3f}",
    ]

    if stage1_ci is not None:
        lines.append(f"- Holdout macro F1 95% CI: {stage1_ci}")
    if stage1_backtest_mean is not None:
        lines.append(f"- Temporal backtest mean macro F1: {stage1_backtest_mean:.3f}")

    lines.extend(
        [
            "",
            "## Stage 2 Fault-Family Study",
            "- Taxonomy: `ground-related`, `phase-to-phase`, `three-phase`, `transformer/internal`, `operational-other`, `unknown/unclassifiable`.",
            f"- Curated distribution events: {len(annotations)}",
            f"- Unknown/unclassifiable events retained for QA only: {unknown_labels}",
            f"- Train/validation events: {stage2_metrics['train_events']}",
            f"- Holdout events: {stage2_metrics['holdout_events']}",
            f"- Selected model: {stage2_metrics['selected_model']}",
            f"- Holdout macro F1: {stage2_selected_metrics['macro_f1']:.3f}",
            f"- Holdout weighted F1: {stage2_selected_metrics['weighted_f1']:.3f}",
        ]
    )
    if stage2_ci is not None:
        lines.append(f"- Holdout macro F1 95% CI: {stage2_ci}")
    if stage2_backtest_mean is not None:
        lines.append(f"- Temporal backtest mean macro F1: {stage2_backtest_mean:.3f}")

    class_merge_mapping = stage2_metrics.get("class_merge_mapping", {})
    if class_merge_mapping:
        lines.extend(["", "## Stage 2 Rare-Class Merge", f"- Applied mapping: `{class_merge_mapping}`"])

    lines.extend(
        [
            "",
            "## Evaluation Upgrades",
            "- Time-aware rolling-origin backtests are included in both stage metric files.",
            "- Bootstrap confidence intervals are reported for model holdout metrics.",
            "- Feature ablation and misclassification exports are written under `outputs/stage1/` and `outputs/stage2/`.",
            "",
            "## Domain Validation Pack",
            "- Expert review templates are written under `outputs/validation/` for event aggregation and stage-2 labels.",
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
    validation_paths = export_domain_validation_pack(root)
    report_markdown = build_markdown_report(root)
    report_path = paths.reports / "summary.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    output_paths = {"report": report_path}
    output_paths.update(figure_paths)
    output_paths.update(validation_paths)
    return output_paths
