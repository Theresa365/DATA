"""Stage-2 draft labeling and annotation helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from scada_faults.config import STAGE2_LABELS
from scada_faults.events import load_events
from scada_faults.paths import ensure_output_dirs

STAGE2_RULES: list[tuple[str, list[str], str]] = [
    (
        "three-phase",
        [r"\b3ph\b", r"three phase", r"three-phase", r"\b3ph t\b", r"\b3ph trip\b"],
        "three-phase keywords",
    ),
    (
        "transformer/internal",
        [
            r"buchholz",
            r"oil surge",
            r"winding temp",
            r"\bwinding\b",
            r"oil temp",
            r"internal fault",
            r"restricted earth fault",
            r"\bref\b",
            r"diff protection operated",
        ],
        "transformer or internal-fault keywords",
    ),
    (
        "ground-related",
        [r"earth fault", r"\be/f\b", r"e/fault", r"to ground", r"\bground\b", r"directional e/f"],
        "earth or ground keywords",
    ),
    (
        "phase-to-phase",
        [
            r"phase a ?& ?b",
            r"phase b ?& ?c",
            r"phase a ?& ?c",
            r"phase b and c",
            r"phase a.?b",
            r"phase b.?c",
            r"phase a.?c",
            r"\bb-c\b",
            r"\ba-c\b",
            r"\ba-b\b",
            r"b-phase",
            r"c-phase",
            r"a-phase",
        ],
        "phase-pair keywords",
    ),
    (
        "operational-other",
        [
            r"under voltage",
            r"undervoltage",
            r"over voltage",
            r"overvoltage",
            r"under frequency",
            r"bus zone",
            r"intertrip",
            r"through fault",
            r"load loss",
            r"did not trip",
            r"smoke",
            r"close block",
            r"overload",
            r"duplicate",
        ],
        "operational or protection-event keywords",
    ),
]


def infer_stage2_label(text_corpus: str) -> tuple[str, str]:
    text = (text_corpus or "").lower()
    evidence: list[str] = []
    for label, patterns, description in STAGE2_RULES:
        matched = [pattern for pattern in patterns if re.search(pattern, text)]
        if matched:
            evidence.append(f"{label}: {description}")
            return label, "; ".join(evidence)
    if "trip" in text or "protection" in text:
        return "operational-other", "generic trip/protection language"
    return "unknown/unclassifiable", "no taxonomy rule matched"


def build_stage2_annotations(events_df: pd.DataFrame) -> pd.DataFrame:
    distribution_events = (
        events_df.loc[events_df["source_name"] == "distribution"]
        .sort_values(["event_date", "fault_no"])
        .reset_index(drop=True)
    )
    labels = distribution_events["text_corpus"].fillna("").map(infer_stage2_label)
    annotation_df = distribution_events[
        [
            "fault_id",
            "fault_no",
            "event_date",
            "trip_time_key",
            "location_primary",
            "system_type",
            "weather",
            "comments_concat",
            "apparatus_concat",
            "is_chrono_anomaly",
            "anomaly_reason",
        ]
    ].copy()
    annotation_df["draft_label"] = labels.map(lambda item: item[0])
    annotation_df["final_label"] = annotation_df["draft_label"]
    annotation_df["evidence"] = labels.map(lambda item: item[1])
    annotation_df["reviewer"] = "codex-initial-curation"
    annotation_df["notes"] = ""
    return annotation_df[
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
            "reviewer",
            "notes",
            "comments_concat",
            "apparatus_concat",
            "is_chrono_anomaly",
            "anomaly_reason",
        ]
    ]


def validate_annotation_labels(annotation_df: pd.DataFrame) -> None:
    allowed = set(STAGE2_LABELS)
    invalid = sorted(set(annotation_df["final_label"].dropna()) - allowed)
    if invalid:
        raise ValueError(f"Invalid stage-2 labels found: {invalid}")


def run_draft_stage2_labels(root: Path | None = None) -> Path:
    events_df = load_events(root, source_name="distribution")
    annotation_df = build_stage2_annotations(events_df)
    validate_annotation_labels(annotation_df)
    paths = ensure_output_dirs(root)
    output_path = paths.annotations / "distribution_stage2_annotations.csv"
    annotation_df.to_csv(output_path, index=False)
    return output_path


def load_stage2_annotations(root: Path | None = None) -> pd.DataFrame:
    paths = ensure_output_dirs(root)
    annotation_path = paths.annotations / "distribution_stage2_annotations.csv"
    if not annotation_path.exists():
        run_draft_stage2_labels(root)
    df = pd.read_csv(annotation_path, parse_dates=["event_date"])
    validate_annotation_labels(df)
    return df
