"""Event-level aggregation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from scada_faults.config import OPERATIONAL_WORST_CASE_ORDER, TEXT_JOIN_TOKEN
from scada_faults.dataset import load_prepared_rows
from scada_faults.paths import ensure_output_dirs
from scada_faults.utils import join_unique, mode_or_unknown, season_from_month


KEYWORD_PATTERNS = {
    "mentions_overcurrent": r"\bo/c\b|overcurrent",
    "mentions_earth_fault": r"earth fault|\be/f\b|ground",
    "mentions_phase": r"\bphase\b",
    "mentions_three_phase": r"\b3ph\b|three phase|three-phase",
    "mentions_buchholz": r"buchholz|oil surge|winding temp|oil temp|internal fault",
    "mentions_diff": r"\bdiff\b|differential",
    "mentions_voltage_issue": r"under voltage|undervoltage|over voltage|overvoltage|under frequency",
    "mentions_trip_failure": r"did not trip|false|no restoration",
}


def worst_case_operational_label(values: list[str]) -> str:
    normalized = [value for value in values if value in OPERATIONAL_WORST_CASE_ORDER]
    if not normalized:
        return "unknown"
    return max(normalized, key=lambda value: OPERATIONAL_WORST_CASE_ORDER[value])


def build_fault_id(group: pd.DataFrame) -> str:
    row = group.iloc[0]
    return f"{row['source_name']}::{row['fault_no']}::{row['event_date_key']}::{row['trip_time_key']}"


def aggregate_events(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["source_name", "fault_no", "event_date_key", "trip_time_key"]
    rows = rows.sort_values(["event_date", "trip_datetime", "fault_no"]).reset_index(drop=True)
    event_records: list[dict[str, object]] = []

    for _, group in rows.groupby(group_cols, dropna=False, sort=True):
        first = group.iloc[0]
        comments_concat = join_unique(group["comments_raw"])
        apparatus_concat = join_unique(group["apparatus_tripped"])
        text_corpus = f"{comments_concat} {apparatus_concat}".strip().lower()
        operational_labels = sorted(set(group["fault_type_clean"].dropna()))
        event_date = group["event_date"].dropna().min() if group["event_date"].notna().any() else pd.NaT
        trip_datetime = group["trip_datetime"].dropna().min() if group["trip_datetime"].notna().any() else pd.NaT
        reclose_delays = group["reclose_delay_hours"].dropna()
        downtimes = group["downtime_hours"].dropna()
        event_record: dict[str, object] = {
            "fault_id": build_fault_id(group),
            "source_name": first["source_name"],
            "fault_no": first["fault_no"],
            "event_date": event_date,
            "trip_datetime": trip_datetime,
            "event_date_key": first["event_date_key"],
            "trip_time_key": first["trip_time_key"],
            "location_primary": mode_or_unknown(group["location_town"]),
            "locations_all": join_unique(group["location_town"]),
            "apparatus_concat": apparatus_concat,
            "comments_concat": comments_concat,
            "text_corpus": text_corpus,
            "system_type": mode_or_unknown(group["system_type_clean"]),
            "substation_area": mode_or_unknown(group["substation_area_clean"]),
            "weather": mode_or_unknown(group["weather_clean"]),
            "reporter": mode_or_unknown(group["reported_by_clean"]),
            "voltage_level_kv": group["voltage_level_kv_num"].dropna().mode().iloc[0]
            if group["voltage_level_kv_num"].dropna().any()
            else pd.NA,
            "voltage_levels_all": join_unique(group["voltage_level_kv"]),
            "row_count": int(len(group)),
            "unique_apparatus_count": int(group["apparatus_tripped"].nunique(dropna=True)),
            "max_downtime_hours": float(downtimes.max()) if not downtimes.empty else 0.0,
            "mean_downtime_hours": float(downtimes.mean()) if not downtimes.empty else 0.0,
            "min_reclose_delay_hours": float(reclose_delays.min()) if not reclose_delays.empty else pd.NA,
            "max_reclose_delay_hours": float(reclose_delays.max()) if not reclose_delays.empty else pd.NA,
            "any_reclosed_clock": bool((group["time_reclosed_status"] == "clock").any()),
            "any_reclosed_arc": bool((group["time_reclosed_status"] == "arc").any()),
            "has_comments": bool(group["comments_raw"].fillna("").str.strip().ne("").any()),
            "mixed_operational_labels": len(operational_labels) > 1,
            "row_operational_labels": TEXT_JOIN_TOKEN.join(operational_labels),
            "stage1_operational_label": worst_case_operational_label(operational_labels),
            "stage1_binary_label": "Permanent"
            if worst_case_operational_label(operational_labels) == "permanent"
            else "Non-permanent",
            "is_chrono_anomaly": bool(group["is_chrono_anomaly"].any()),
            "anomaly_reason": join_unique(group["anomaly_reason"]),
        }

        if pd.isna(trip_datetime) and not pd.isna(event_date):
            event_record["event_hour"] = -1
            event_record["event_weekday"] = int(event_date.dayofweek)
            event_record["event_month"] = int(event_date.month)
        elif not pd.isna(trip_datetime):
            event_record["event_hour"] = int(trip_datetime.hour)
            event_record["event_weekday"] = int(trip_datetime.dayofweek)
            event_record["event_month"] = int(trip_datetime.month)
        else:
            event_record["event_hour"] = -1
            event_record["event_weekday"] = -1
            event_record["event_month"] = -1
        event_record["season"] = season_from_month(event_record["event_month"])

        for feature_name, pattern in KEYWORD_PATTERNS.items():
            event_record[feature_name] = int(bool(re.search(pattern, text_corpus)))

        event_records.append(event_record)

    events_df = pd.DataFrame(event_records).sort_values(["event_date", "fault_no"]).reset_index(drop=True)
    return events_df


def build_event_tables(root: Path | None = None) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for source_name in ["distribution", "system"]:
        rows = load_prepared_rows(root, source_name=source_name)
        tables[source_name] = aggregate_events(rows)
    return tables


def save_event_tables(tables: dict[str, pd.DataFrame], root: Path | None = None) -> dict[str, Path]:
    paths = ensure_output_dirs(root)
    written: dict[str, Path] = {}
    summary: dict[str, dict[str, object]] = {}
    for source_name, table in tables.items():
        output_path = paths.events / f"{source_name}_events.csv"
        table.to_csv(output_path, index=False)
        written[source_name] = output_path
        summary[source_name] = {
            "events": int(len(table)),
            "chrono_anomalies": int(table["is_chrono_anomaly"].sum()),
            "stage1_label_counts": table["stage1_operational_label"].value_counts(dropna=False).to_dict(),
        }
    (paths.events / "event_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return written


def run_build_events(root: Path | None = None) -> dict[str, Path]:
    tables = build_event_tables(root)
    return save_event_tables(tables, root=root)


def load_events(root: Path | None = None, source_name: str = "distribution") -> pd.DataFrame:
    paths = ensure_output_dirs(root)
    event_path = paths.events / f"{source_name}_events.csv"
    if not event_path.exists():
        run_build_events(root)
    return pd.read_csv(event_path, parse_dates=["event_date", "trip_datetime"])
