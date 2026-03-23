"""Domain-expert review exports for event aggregation and stage-2 labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scada_faults.curation import load_stage2_annotations
from scada_faults.events import load_events
from scada_faults.paths import ensure_output_dirs


def build_event_aggregation_review(events_df: pd.DataFrame) -> pd.DataFrame:
    review_df = events_df.loc[events_df["row_count"] > 1].copy()
    review_df = review_df.sort_values(["row_count", "event_date", "fault_no"], ascending=[False, True, True])
    review_df["review_status"] = "pending"
    review_df["domain_expert_decision"] = ""
    review_df["domain_expert_notes"] = ""
    return review_df[
        [
            "fault_id",
            "fault_no",
            "event_date",
            "trip_time_key",
            "row_count",
            "unique_apparatus_count",
            "location_primary",
            "locations_all",
            "substation_area",
            "weather",
            "apparatus_concat",
            "comments_concat",
            "mixed_operational_labels",
            "row_operational_labels",
            "is_chrono_anomaly",
            "anomaly_reason",
            "review_status",
            "domain_expert_decision",
            "domain_expert_notes",
        ]
    ].reset_index(drop=True)


def build_stage2_label_review(annotation_df: pd.DataFrame) -> pd.DataFrame:
    review_df = annotation_df.copy().sort_values(["event_date", "fault_no"]).reset_index(drop=True)
    review_df["review_status"] = "pending"
    review_df["domain_expert_label"] = ""
    review_df["domain_expert_notes"] = ""
    return review_df[
        [
            "fault_id",
            "fault_no",
            "event_date",
            "trip_time_key",
            "location_primary",
            "system_type",
            "weather",
            "draft_label",
            "final_label",
            "evidence",
            "comments_concat",
            "apparatus_concat",
            "is_chrono_anomaly",
            "anomaly_reason",
            "review_status",
            "domain_expert_label",
            "domain_expert_notes",
        ]
    ]


def build_validation_guide() -> str:
    return "\n".join(
        [
            "# Domain Validation Guide",
            "",
            "This folder supports expert review of two high-risk steps in the pipeline.",
            "",
            "## 1. Event Aggregation Review",
            "- File: `event_aggregation_review.csv`",
            "- Purpose: confirm that apparatus-level rows grouped into one `fault_id` truly belong to the same operational event.",
            "- Focus on high `row_count`, mixed apparatus, mixed operational labels, and chronology anomalies.",
            "- Fill in `review_status`, `domain_expert_decision`, and `domain_expert_notes`.",
            "",
            "## 2. Stage-2 Label Review",
            "- File: `stage2_label_review.csv`",
            "- Purpose: verify whether the automatically drafted electrical fault-family label matches engineering interpretation.",
            "- Compare `draft_label`, `final_label`, `evidence`, `comments_concat`, and `apparatus_concat`.",
            "- Fill in `review_status`, `domain_expert_label`, and `domain_expert_notes`.",
            "",
            "## Suggested Review Outcomes",
            "- `accepted`: current pipeline decision looks correct",
            "- `revise`: label or aggregation should change",
            "- `uncertain`: insufficient evidence from SCADA text alone",
            "",
            "## Recommendation",
            "- Prioritize events with `row_count > 2`, `mixed_operational_labels = True`, or `unknown/unclassifiable` labels first.",
        ]
    ) + "\n"


def export_domain_validation_pack(root: Path | None = None) -> dict[str, Path]:
    paths = ensure_output_dirs(root)
    distribution_events = load_events(root, source_name="distribution")
    annotations = load_stage2_annotations(root)

    aggregation_path = paths.validation / "event_aggregation_review.csv"
    stage2_review_path = paths.validation / "stage2_label_review.csv"
    guide_path = paths.validation / "README.md"

    build_event_aggregation_review(distribution_events).to_csv(aggregation_path, index=False)
    build_stage2_label_review(annotations).to_csv(stage2_review_path, index=False)
    guide_path.write_text(build_validation_guide(), encoding="utf-8")

    return {
        "event_aggregation_review": aggregation_path,
        "stage2_label_review": stage2_review_path,
        "validation_guide": guide_path,
    }
