"""Model training, evaluation, and artifact generation."""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from scada_faults.config import DEFAULT_RANDOM_STATE, STAGE2_LABELS
from scada_faults.curation import load_stage2_annotations
from scada_faults.events import load_events
from scada_faults.paths import ensure_output_dirs

NUMERIC_FEATURES = [
    "event_hour",
    "event_weekday",
    "event_month",
    "row_count",
    "unique_apparatus_count",
    "max_downtime_hours",
    "mean_downtime_hours",
    "min_reclose_delay_hours",
    "max_reclose_delay_hours",
    "voltage_level_kv",
    "any_reclosed_clock",
    "any_reclosed_arc",
    "has_comments",
    "mentions_overcurrent",
    "mentions_earth_fault",
    "mentions_phase",
    "mentions_three_phase",
    "mentions_buchholz",
    "mentions_diff",
    "mentions_voltage_issue",
    "mentions_trip_failure",
]

CATEGORICAL_FEATURES = [
    "substation_area",
    "weather",
    "reporter",
    "season",
    "location_primary",
]

TEXT_FEATURES = ["comments_concat", "apparatus_concat"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES


@dataclass
class TrainableModelSpec:
    name: str
    estimator_factory: Callable[[], Pipeline]


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("comments", TfidfVectorizer(max_features=80, ngram_range=(1, 2)), "comments_concat"),
            ("apparatus", TfidfVectorizer(max_features=60, ngram_range=(1, 2)), "apparatus_concat"),
        ]
    )


def _make_classifier_pipeline(classifier) -> Pipeline:
    return Pipeline(steps=[("preprocessor", _make_preprocessor()), ("classifier", classifier)])


def stage1_model_specs() -> list[TrainableModelSpec]:
    return [
        TrainableModelSpec(
            name="logistic-regression",
            estimator_factory=lambda: _make_classifier_pipeline(
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=DEFAULT_RANDOM_STATE,
                )
            ),
        ),
        TrainableModelSpec(
            name="decision-tree",
            estimator_factory=lambda: _make_classifier_pipeline(
                DecisionTreeClassifier(
                    class_weight="balanced",
                    max_depth=5,
                    min_samples_leaf=2,
                    random_state=DEFAULT_RANDOM_STATE,
                )
            ),
        ),
        TrainableModelSpec(
            name="random-forest",
            estimator_factory=lambda: _make_classifier_pipeline(
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_estimators=300,
                    random_state=DEFAULT_RANDOM_STATE,
                )
            ),
        ),
    ]


def stage2_model_specs() -> list[TrainableModelSpec]:
    return [
        TrainableModelSpec(
            name="logistic-regression",
            estimator_factory=lambda: _make_classifier_pipeline(
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=DEFAULT_RANDOM_STATE,
                )
            ),
        ),
        TrainableModelSpec(
            name="random-forest",
            estimator_factory=lambda: _make_classifier_pipeline(
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_estimators=400,
                    random_state=DEFAULT_RANDOM_STATE,
                )
            ),
        ),
    ]


def prepare_model_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.copy()
    df = df.loc[~df["is_chrono_anomaly"].fillna(False)].sort_values(["event_date", "fault_id"]).reset_index(drop=True)
    for column in NUMERIC_FEATURES:
        if column in {"any_reclosed_clock", "any_reclosed_arc", "has_comments"}:
            df[column] = df[column].fillna(False).astype(int)
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("unknown").astype(str)
    for column in TEXT_FEATURES:
        df[column] = df[column].fillna("").astype(str)
    return df


def chronological_holdout_split(df: pd.DataFrame, holdout_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 8:
        raise ValueError("Not enough events for chronological holdout.")
    holdout_size = max(1, math.ceil(len(df) * holdout_fraction))
    train_val = df.iloc[:-holdout_size].reset_index(drop=True)
    holdout = df.iloc[-holdout_size:].reset_index(drop=True)
    return train_val, holdout


def rolling_origin_splits(y: pd.Series, max_splits: int = 3, minimum_train: int = 12) -> list[tuple[np.ndarray, np.ndarray]]:
    required_classes = min(2, y.nunique())
    seen_classes: set[str] = set()
    first_all_classes_index = 0
    for index, value in enumerate(y):
        seen_classes.add(str(value))
        if len(seen_classes) >= required_classes:
            first_all_classes_index = index + 1
            break
    min_train_size = max(minimum_train, first_all_classes_index)
    n_samples = len(y)
    if n_samples <= min_train_size + 1:
        split_point = max(required_classes, n_samples - 1)
        return [(np.arange(split_point), np.arange(split_point, n_samples))]

    remaining = n_samples - min_train_size
    val_size = max(1, remaining // max_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train_size
    while train_end < n_samples and len(folds) < max_splits:
        val_end = min(n_samples, train_end + val_size)
        if val_end <= train_end:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        if len(pd.Series(y.iloc[train_idx]).unique()) >= required_classes:
            folds.append((train_idx, val_idx))
        train_end = val_end
    if not folds:
        split_point = max(required_classes, n_samples - 1)
        folds.append((np.arange(split_point), np.arange(split_point, n_samples)))
    return folds


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(label_precision),
                "recall": float(label_recall),
                "f1": float(label_f1),
                "support": int(label_support),
            }
            for label, label_precision, label_recall, label_f1, label_support in zip(
                labels, precision, recall, f1, support, strict=False
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return metrics


def stage1_rule_baseline(df: pd.DataFrame) -> pd.Series:
    rule_non_permanent = (
        (df["max_downtime_hours"] <= 0.25)
        | (df["min_reclose_delay_hours"].fillna(999) <= 0.25)
        | (df["any_reclosed_arc"] == 1)
        | (df["mentions_trip_failure"] == 1)
    )
    return pd.Series(np.where(rule_non_permanent, "Non-permanent", "Permanent"), index=df.index)


def majority_baseline(y_train: pd.Series, n: int) -> pd.Series:
    majority_label = y_train.mode().iloc[0]
    return pd.Series([majority_label] * n)


def choose_binary_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, np.ndarray]:
    candidate_thresholds = np.arange(0.30, 0.71, 0.05)
    best_threshold = 0.50
    best_predictions = (scores >= best_threshold).astype(int)
    best_score = -1.0
    mapped_true = (y_true == "Permanent").astype(int)
    for threshold in candidate_thresholds:
        predictions = (scores >= threshold).astype(int)
        score = f1_score(mapped_true, predictions, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_predictions = predictions
    labels = np.where(best_predictions == 1, "Permanent", "Non-permanent")
    return best_threshold, labels


def evaluate_trainable_model(
    spec: TrainableModelSpec,
    train_val_df: pd.DataFrame,
    target_column: str,
    labels: list[str],
    *,
    binary_task: bool,
) -> tuple[dict[str, object], Pipeline, float]:
    X = train_val_df[MODEL_FEATURES]
    y = train_val_df[target_column]
    fold_rows: list[dict[str, object]] = []
    thresholds: list[float] = []

    for fold_number, (train_idx, val_idx) in enumerate(rolling_origin_splits(y), start=1):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < min(2, y.nunique()):
            continue
        estimator = spec.estimator_factory()
        estimator.fit(X.iloc[train_idx], y_train)
        if binary_task:
            probabilities = estimator.predict_proba(X.iloc[val_idx])
            positive_index = list(estimator.classes_).index("Permanent")
            threshold, y_pred = choose_binary_threshold(y.iloc[val_idx], probabilities[:, positive_index])
            thresholds.append(threshold)
        else:
            y_pred = estimator.predict(X.iloc[val_idx])
        metrics = compute_metrics(y.iloc[val_idx], pd.Series(y_pred, index=y.iloc[val_idx].index), labels=labels)
        metrics["fold"] = fold_number
        fold_rows.append(metrics)

    if not fold_rows:
        raise ValueError(f"No valid rolling-origin folds were available for model '{spec.name}'.")

    mean_macro_f1 = float(np.mean([row["macro_f1"] for row in fold_rows]))
    mean_weighted_f1 = float(np.mean([row["weighted_f1"] for row in fold_rows]))
    chosen_threshold = float(np.mean(thresholds)) if thresholds else 0.50
    final_model = spec.estimator_factory()
    final_model.fit(X, y)
    return (
        {
            "model_name": spec.name,
            "cv_macro_f1": mean_macro_f1,
            "cv_weighted_f1": mean_weighted_f1,
            "cv_folds": fold_rows,
            "threshold": chosen_threshold,
        },
        final_model,
        chosen_threshold,
    )


def save_confusion_matrix_figure(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, matrix[row_index, col_index], ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_permutation_importance(
    model: Pipeline,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    importance = permutation_importance(
        model,
        X_holdout,
        y_holdout,
        n_repeats=25,
        random_state=DEFAULT_RANDOM_STATE,
        scoring="f1_weighted",
    )
    importance_df = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output_path, index=False)
    return importance_df


def maybe_write_shap_note(output_path: Path) -> None:
    message = (
        "SHAP output not generated. Install the optional 'extras' dependencies and rerun training to enable it."
        if importlib.util.find_spec("shap") is None
        else "SHAP is installed, but this project currently reports permutation importance as the default explanation."
    )
    output_path.write_text(message, encoding="utf-8")


def run_stage1_training(root: Path | None = None) -> dict[str, Path]:
    paths = ensure_output_dirs(root)
    events_df = load_events(root, source_name="distribution")
    model_df = prepare_model_frame(events_df)
    train_val_df, holdout_df = chronological_holdout_split(model_df)
    labels = ["Permanent", "Non-permanent"]

    baseline_predictions = {
        "majority-baseline": majority_baseline(train_val_df["stage1_binary_label"], len(holdout_df)),
        "rule-baseline": stage1_rule_baseline(holdout_df),
    }

    trainable_results: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}

    for spec in stage1_model_specs():
        result, model, threshold = evaluate_trainable_model(
            spec,
            train_val_df,
            target_column="stage1_binary_label",
            labels=labels,
            binary_task=True,
        )
        trainable_results.append(result)
        fitted_models[spec.name] = model
        thresholds[spec.name] = threshold

    selected_result = max(trainable_results, key=lambda item: item["cv_macro_f1"])
    selected_model_name = str(selected_result["model_name"])
    selected_model = fitted_models[selected_model_name]
    selected_threshold = thresholds[selected_model_name]

    holdout_predictions: dict[str, pd.Series] = {}
    for model_name, model in fitted_models.items():
        probabilities = model.predict_proba(holdout_df[MODEL_FEATURES])
        positive_index = list(model.classes_).index("Permanent")
        threshold = thresholds[model_name]
        labels_pred = np.where(probabilities[:, positive_index] >= threshold, "Permanent", "Non-permanent")
        holdout_predictions[model_name] = pd.Series(labels_pred, index=holdout_df.index)

    evaluation_summary: dict[str, object] = {
        "selected_model": selected_model_name,
        "selected_threshold": selected_threshold,
        "train_events": int(len(train_val_df)),
        "holdout_events": int(len(holdout_df)),
        "chronology": {
            "train_start": str(train_val_df["event_date"].min().date()),
            "train_end": str(train_val_df["event_date"].max().date()),
            "holdout_start": str(holdout_df["event_date"].min().date()),
            "holdout_end": str(holdout_df["event_date"].max().date()),
        },
        "comparisons": {},
        "cv_results": trainable_results,
    }

    y_holdout = holdout_df["stage1_binary_label"]
    for baseline_name, predictions in baseline_predictions.items():
        evaluation_summary["comparisons"][baseline_name] = compute_metrics(y_holdout, predictions, labels)
    for model_name, predictions in holdout_predictions.items():
        evaluation_summary["comparisons"][model_name] = compute_metrics(y_holdout, predictions, labels)

    predictions_df = holdout_df[["fault_id", "fault_no", "event_date", "location_primary", "stage1_binary_label"]].copy()
    predictions_df = predictions_df.rename(columns={"stage1_binary_label": "actual_label"})
    for model_name, predictions in baseline_predictions.items():
        predictions_df[model_name] = predictions.values
    for model_name, predictions in holdout_predictions.items():
        predictions_df[model_name] = predictions.values

    metrics_path = paths.stage1 / "stage1_metrics.json"
    predictions_path = paths.stage1 / "stage1_holdout_predictions.csv"
    model_path = paths.stage1 / f"{selected_model_name}.joblib"
    importance_path = paths.stage1 / "stage1_permutation_importance.csv"
    confusion_path = paths.figures / "stage1_confusion_matrix.png"
    shap_note_path = paths.stage1 / "stage1_shap_note.txt"

    metrics_path.write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False)
    joblib.dump(selected_model, model_path)
    save_permutation_importance(selected_model, holdout_df[MODEL_FEATURES], y_holdout, importance_path)
    save_confusion_matrix_figure(
        y_holdout,
        holdout_predictions[selected_model_name],
        labels,
        confusion_path,
        "Stage 1: Permanent vs Non-permanent",
    )
    maybe_write_shap_note(shap_note_path)

    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "model": model_path,
        "importance": importance_path,
        "confusion_matrix": confusion_path,
    }


def merge_rare_stage2_classes(
    train_val_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    threshold: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    counts = train_val_df["stage2_label"].value_counts()
    all_labels = sorted(set(train_val_df["stage2_label"]) | set(holdout_df["stage2_label"]))
    rare_classes = {label for label in all_labels if counts.get(label, 0) < threshold}
    rare_classes.discard("operational-other")
    mapping = {label: "operational-other" for label in rare_classes}
    if not mapping:
        return train_val_df, holdout_df, {}
    updated_train = train_val_df.copy()
    updated_holdout = holdout_df.copy()
    updated_train["stage2_label"] = updated_train["stage2_label"].replace(mapping)
    updated_holdout["stage2_label"] = updated_holdout["stage2_label"].replace(mapping)
    return updated_train, updated_holdout, mapping


def run_stage2_training(root: Path | None = None) -> dict[str, Path]:
    paths = ensure_output_dirs(root)
    events_df = load_events(root, source_name="distribution")
    model_df = prepare_model_frame(events_df)
    annotations = load_stage2_annotations(root)
    merged = model_df.merge(annotations[["fault_id", "final_label"]], on="fault_id", how="left")
    merged = merged.rename(columns={"final_label": "stage2_label"})
    merged = merged.loc[merged["stage2_label"].notna()].copy()
    merged = merged.loc[merged["stage2_label"] != "unknown/unclassifiable"].reset_index(drop=True)
    train_val_df, holdout_df = chronological_holdout_split(merged)
    train_val_df, holdout_df, rare_merge_mapping = merge_rare_stage2_classes(train_val_df, holdout_df)
    labels = sorted(train_val_df["stage2_label"].unique())

    baseline_predictions = {
        "majority-baseline": majority_baseline(train_val_df["stage2_label"], len(holdout_df)),
    }

    trainable_results: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}

    for spec in stage2_model_specs():
        result, model, _ = evaluate_trainable_model(
            spec,
            train_val_df,
            target_column="stage2_label",
            labels=labels,
            binary_task=False,
        )
        trainable_results.append(result)
        fitted_models[spec.name] = model

    selected_result = max(trainable_results, key=lambda item: item["cv_macro_f1"])
    selected_model_name = str(selected_result["model_name"])
    selected_model = fitted_models[selected_model_name]

    y_holdout = holdout_df["stage2_label"]
    holdout_predictions = {
        model_name: pd.Series(model.predict(holdout_df[MODEL_FEATURES]), index=holdout_df.index)
        for model_name, model in fitted_models.items()
    }

    evaluation_summary: dict[str, object] = {
        "selected_model": selected_model_name,
        "train_events": int(len(train_val_df)),
        "holdout_events": int(len(holdout_df)),
        "class_merge_mapping": rare_merge_mapping,
        "comparisons": {},
        "cv_results": trainable_results,
    }
    for baseline_name, predictions in baseline_predictions.items():
        evaluation_summary["comparisons"][baseline_name] = compute_metrics(y_holdout, predictions, labels)
    for model_name, predictions in holdout_predictions.items():
        evaluation_summary["comparisons"][model_name] = compute_metrics(y_holdout, predictions, labels)

    predictions_df = holdout_df[["fault_id", "fault_no", "event_date", "location_primary", "stage2_label"]].copy()
    predictions_df = predictions_df.rename(columns={"stage2_label": "actual_label"})
    for baseline_name, predictions in baseline_predictions.items():
        predictions_df[baseline_name] = predictions.values
    for model_name, predictions in holdout_predictions.items():
        predictions_df[model_name] = predictions.values

    metrics_path = paths.stage2 / "stage2_metrics.json"
    predictions_path = paths.stage2 / "stage2_holdout_predictions.csv"
    model_path = paths.stage2 / f"{selected_model_name}.joblib"
    importance_path = paths.stage2 / "stage2_permutation_importance.csv"
    confusion_path = paths.figures / "stage2_confusion_matrix.png"
    shap_note_path = paths.stage2 / "stage2_shap_note.txt"

    metrics_path.write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False)
    joblib.dump(selected_model, model_path)
    save_permutation_importance(selected_model, holdout_df[MODEL_FEATURES], y_holdout, importance_path)
    save_confusion_matrix_figure(
        y_holdout,
        holdout_predictions[selected_model_name],
        labels,
        confusion_path,
        "Stage 2: Electrical fault family",
    )
    maybe_write_shap_note(shap_note_path)
    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "model": model_path,
        "importance": importance_path,
        "confusion_matrix": confusion_path,
    }
