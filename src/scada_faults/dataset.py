"""Raw workbook ingestion and row-level normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scada_faults.config import RAW_COLUMN_MAP, ROW_ID_COLUMNS
from scada_faults.paths import discover_raw_workbooks, ensure_output_dirs
from scada_faults.utils import (
    canonical_lower,
    combine_date_and_time,
    combine_reclose_datetime,
    infer_anomaly_reason,
    normalize_reporter,
    normalize_string,
    normalize_weather,
    parse_clock_time,
    parse_float,
    parse_mixed_date,
)


@dataclass
class PreparationResult:
    source_name: str
    rows: pd.DataFrame
    summary: dict[str, object]


def read_raw_workbook(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = df.loc[:, [col for col in df.columns if str(col) in RAW_COLUMN_MAP]]
    df = df.rename(columns=RAW_COLUMN_MAP)
    for column in ROW_ID_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df["source_name"] = source_name
    df["source_file"] = path.name
    return df


def normalize_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    raw_columns = ROW_ID_COLUMNS + ["source_name", "source_file"]
    df = df.drop_duplicates(subset=raw_columns).reset_index(drop=True)

    df["fault_no"] = df["fault_no"].map(normalize_string)
    df["date_text"] = df["date_raw"].map(normalize_string)
    df["event_date"] = df["date_text"].map(parse_mixed_date)
    df["trip_time_text"] = df["trip_time_raw"].map(normalize_string)
    trip_parsed = df["trip_time_raw"].map(parse_clock_time)
    df["trip_time_parsed"] = trip_parsed.map(lambda item: item[0])
    df["trip_time_status"] = trip_parsed.map(lambda item: item[1])

    reclose_parsed = df["time_reclosed_raw"].map(parse_clock_time)
    df["time_reclosed_text"] = df["time_reclosed_raw"].map(normalize_string)
    df["time_reclosed_parsed"] = reclose_parsed.map(lambda item: item[0])
    df["time_reclosed_status"] = reclose_parsed.map(lambda item: item[1])

    df["trip_datetime"] = [
        combine_date_and_time(event_date, trip_time)
        for event_date, trip_time in zip(df["event_date"], df["trip_time_parsed"], strict=False)
    ]
    df["reclose_datetime"] = [
        combine_reclose_datetime(trip_dt, event_date, reclose_time)
        for trip_dt, event_date, reclose_time in zip(
            df["trip_datetime"], df["event_date"], df["time_reclosed_parsed"], strict=False
        )
    ]

    df["reclose_delay_hours"] = (
        (df["reclose_datetime"] - df["trip_datetime"]).dt.total_seconds() / 3600.0
    )
    df["downtime_hours"] = df["downtime_hours_raw"].map(parse_float)
    df["voltage_level_kv_num"] = df["voltage_level_kv"].map(parse_float)

    df["weather_clean"] = df["weather_raw"].map(normalize_weather)
    df["reported_by_clean"] = df["reported_by_raw"].map(normalize_reporter)
    df["comments_clean"] = df["comments_raw"].map(canonical_lower)
    df["apparatus_clean"] = df["apparatus_tripped"].map(canonical_lower)
    df["location_clean"] = df["location_town"].map(canonical_lower)
    df["system_type_clean"] = df["system_type"].map(normalize_string)
    df["substation_area_clean"] = df["substation_area"].map(normalize_string)
    df["severity_clean"] = df["severity_raw"].map(canonical_lower)
    df["fault_type_clean"] = df["fault_type_raw"].map(canonical_lower)

    df["event_date_key"] = df["event_date"].dt.strftime("%Y-%m-%d").fillna("missing")
    df["trip_time_key"] = df["trip_time_parsed"].dt.strftime("%H:%M").fillna("missing")
    df["is_chrono_anomaly"] = [
        bool(infer_anomaly_reason(fault_no or "", event_date))
        for fault_no, event_date in zip(df["fault_no"], df["event_date"], strict=False)
    ]
    df["anomaly_reason"] = [
        infer_anomaly_reason(fault_no or "", event_date)
        for fault_no, event_date in zip(df["fault_no"], df["event_date"], strict=False)
    ]
    df["event_year"] = df["event_date"].dt.year
    df["event_month"] = df["event_date"].dt.month
    return df


def build_preparation_summary(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "unique_fault_numbers": int(df["fault_no"].nunique(dropna=True)),
        "chrono_anomaly_rows": int(df["is_chrono_anomaly"].sum()),
        "missing_trip_time_rows": int((df["trip_time_status"] != "clock").sum()),
        "missing_reclose_rows": int((df["time_reclosed_status"] == "missing").sum()),
        "weather_values": df["weather_clean"].value_counts(dropna=False).to_dict(),
    }


def prepare_workbook(path: Path, source_name: str) -> PreparationResult:
    raw_df = read_raw_workbook(path, source_name=source_name)
    rows = normalize_rows(raw_df)
    return PreparationResult(source_name=source_name, rows=rows, summary=build_preparation_summary(rows))


def prepare_all_sources(root: Path | None = None) -> dict[str, PreparationResult]:
    raw_workbooks = discover_raw_workbooks(root)
    return {
        source_name: prepare_workbook(path, source_name=source_name)
        for source_name, path in raw_workbooks.items()
    }


def save_preparation_results(results: dict[str, PreparationResult], root: Path | None = None) -> dict[str, Path]:
    paths = ensure_output_dirs(root)
    written: dict[str, Path] = {}
    for source_name, result in results.items():
        rows_path = paths.prepared / f"{source_name}_rows.csv"
        summary_path = paths.prepared / f"{source_name}_summary.json"
        result.rows.to_csv(rows_path, index=False)
        summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
        written[source_name] = rows_path
    return written


def run_prepare_data(root: Path | None = None) -> dict[str, Path]:
    results = prepare_all_sources(root)
    return save_preparation_results(results, root=root)


def load_prepared_rows(root: Path | None = None, source_name: str = "distribution") -> pd.DataFrame:
    paths = ensure_output_dirs(root)
    rows_path = paths.prepared / f"{source_name}_rows.csv"
    if not rows_path.exists():
        run_prepare_data(root)
    df = pd.read_csv(
        rows_path,
        parse_dates=["event_date", "trip_time_parsed", "time_reclosed_parsed", "trip_datetime", "reclose_datetime"],
    )
    return df
