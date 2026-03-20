from __future__ import annotations

from pathlib import Path

import pandas as pd

from scada_faults.curation import load_stage2_annotations, validate_annotation_labels
from scada_faults.dataset import run_prepare_data
from scada_faults.events import load_events, run_build_events
from scada_faults.modeling import chronological_holdout_split, run_stage1_training, run_stage2_training
from scada_faults.reports import run_report_results


RAW_COLUMNS = [
    "Date",
    "Trip Time",
    "Fault No",
    "Location (Town)",
    "Apparatus Tripped",
    "Voltage Level(KV)",
    "System Type",
    "Substation Area",
    "Weather",
    "Time Reclosed",
    "Downtime (Hours)",
    "Reported By",
    "Comments",
    "Month",
    "Severity",
    "Fault type",
]


def _build_row(
    date_text: str,
    trip_time: str,
    fault_no: str,
    apparatus: str,
    comments: str,
    fault_type: str,
    severity: str,
    *,
    location: str = "Gaborone",
    voltage: int = 11,
    system_type: str = "Distribution",
    area: str = "South",
    weather: str = "Clear",
    time_reclosed: str = "08:45",
    downtime: float = 0.5,
    reporter: str = "SCADA/Test",
) -> dict[str, object]:
    return {
        "Date": date_text,
        "Trip Time": trip_time,
        "Fault No": fault_no,
        "Location (Town)": location,
        "Apparatus Tripped": apparatus,
        "Voltage Level(KV)": voltage,
        "System Type": system_type,
        "Substation Area": area,
        "Weather": weather,
        "Time Reclosed": time_reclosed,
        "Downtime (Hours)": downtime,
        "Reported By": reporter,
        "Comments": comments,
        "Month": pd.to_datetime(date_text, dayfirst=True).month_name(),
        "Severity": severity,
        "Fault type": fault_type,
    }


def _write_synthetic_workbooks(root: Path) -> None:
    distribution_rows: list[dict[str, object]] = []
    system_rows: list[dict[str, object]] = []
    base_date = pd.Timestamp("2023-01-01")
    categories = [
        ("ground-related", "Earth Fault (E/F) protection trip", "Permanent", "High", "NIL", 5.0),
        ("operational-other", "Bus zone trip", "Transient", "Low", "08:40", 0.1),
        ("transformer/internal", "Buchholz trip, Oil surge, internal fault suspected", "Permanent", "High", "NIL", 7.0),
    ]

    for index in range(24):
        date_text = (base_date + pd.Timedelta(days=index * 21)).strftime("%d/%m/%Y")
        trip_time = f"{6 + (index % 10):02d}:{(index * 7) % 60:02d}"
        fault_no = f"{index + 1:03d}/23"
        label_name, comments, fault_type, severity, time_reclosed, downtime = categories[index % len(categories)]
        apparatus = f"Station {index % 4} Transformer {index % 3} LV CB {index % 5}H0"
        distribution_rows.append(
            _build_row(
                date_text,
                trip_time,
                fault_no,
                apparatus,
                comments,
                fault_type,
                severity,
                time_reclosed=time_reclosed,
                downtime=downtime,
            )
        )
        if index % 6 == 0:
            distribution_rows.append(
                _build_row(
                    date_text,
                    trip_time,
                    fault_no,
                    f"{apparatus} AUX",
                    "DID NOT TRIP",
                    "Transient",
                    "Low",
                    time_reclosed="NIL",
                    downtime=0.0,
                )
            )
        system_rows.extend(distribution_rows[-2:] if index % 6 == 0 else distribution_rows[-1:])

    for index in range(3):
        date_text = (base_date + pd.Timedelta(days=index * 30)).strftime("%d/%m/%Y")
        system_rows.append(
            _build_row(
                date_text,
                "10:10",
                f"T{index + 1:03d}/23",
                f"Transmission Line {index}",
                "3Ph T + L/O",
                "Permanent",
                "High",
                system_type="Transmission",
                voltage=132,
                time_reclosed="NIL",
                downtime=4.0,
            )
        )

    distribution_df = pd.DataFrame(distribution_rows, columns=RAW_COLUMNS)
    system_df = pd.DataFrame(system_rows, columns=RAW_COLUMNS)
    distribution_path = root / "Monthly Fault Data Analysis Jan2023_June2025 Distribution only (1).xlsx"
    system_path = root / "Monthly Fault Data Analysis Jan2023_June2025 system type (version 1).xlsb (1).xlsx"
    distribution_df.to_excel(distribution_path, index=False)
    system_df.to_excel(system_path, index=False)


def test_end_to_end_pipeline_and_leakage(tmp_path: Path) -> None:
    _write_synthetic_workbooks(tmp_path)

    prepare_outputs = run_prepare_data(tmp_path)
    build_outputs = run_build_events(tmp_path)
    stage1_outputs = run_stage1_training(tmp_path)
    stage2_outputs = run_stage2_training(tmp_path)
    report_outputs = run_report_results(tmp_path)

    for output in [
        *prepare_outputs.values(),
        *build_outputs.values(),
        *stage1_outputs.values(),
        *stage2_outputs.values(),
        *report_outputs.values(),
    ]:
        assert Path(output).exists()

    annotations = load_stage2_annotations(tmp_path)
    validate_annotation_labels(annotations)

    events = load_events(tmp_path, source_name="distribution")
    clean_events = events.loc[~events["is_chrono_anomaly"].fillna(False)].reset_index(drop=True)
    train_val, holdout = chronological_holdout_split(clean_events)
    assert set(train_val["fault_id"]).isdisjoint(set(holdout["fault_id"]))
