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
FEATURE_SET_REGISTRY = {
    "all_features": MODEL_FEATURES,
    "structured_only": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    "text_only": TEXT_FEATURES,
    "no_text": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    "no_numeric": CATEGORICAL_FEATURES + TEXT_FEATURES,
    "no_categorical": NUMERIC_FEATURES + TEXT_FEATURES,
}


@dataclass
class TrainableModelSpec:
    name: str
    estimator_factory: Callable[[], Pipeline]


def _selected_features(feature_names: list[str], selected: list[str]) -> list[str]:
    return [feature_name for feature_name in feature_names if feature_name in selected]


def _make_preprocessor(selected_features: list[str] | None = None) -> ColumnTransformer:
    feature_selection = selected_features or MODEL_FEATURES
    transformers: list[tuple[str, object, object]] = []

    numeric_features = _selected_features(NUMERIC_FEATURES, feature_selection)
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_features,
            )
        )

    categorical_features = _selected_features(CATEGORICAL_FEATURES, feature_selection)
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    if "comments_concat" in feature_selection:
        transformers.append(("comments", TfidfVectorizer(max_features=80, ngram_range=(1, 2)), "comments_concat"))
    if "apparatus_concat" in feature_selection:
        transformers.append(("apparatus", TfidfVectorizer(max_features=60, ngram_range=(1, 2)), "apparatus_concat"))

    return ColumnTransformer(transformers=transformers)


def _make_classifier_pipeline(classifier, selected_features: list[str] | None = None) -> Pipeline:
    return Pipeline(steps=[("preprocessor", _make_preprocessor(selected_features)), ("classifier", classifier)])


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


def bootstrap_metric_confidence_intervals(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length for bootstrap intervals.")

    rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    alpha = (1.0 - confidence_level) / 2.0
    metric_samples = {"accuracy": [], "macro_f1": [], "weighted_f1": []}

    for _ in range(n_bootstrap):
        sampled_idx = rng.integers(0, len(y_true_array), len(y_true_array))
        sampled_true = pd.Series(y_true_array[sampled_idx])
        sampled_pred = pd.Series(y_pred_array[sampled_idx])
        metrics = compute_metrics(sampled_true, sampled_pred, labels)
        for metric_name in metric_samples:
            metric_samples[metric_name].append(float(metrics[metric_name]))

    return {
        metric_name: {
            "lower": float(np.quantile(samples, alpha)),
            "median": float(np.quantile(samples, 0.5)),
            "upper": float(np.quantile(samples, 1.0 - alpha)),
        }
        for metric_name, samples in metric_samples.items()
    }


def summarize_temporal_backtest(fold_metrics: list[dict[str, object]]) -> dict[str, object]:
    metric_names = ["accuracy", "macro_f1", "weighted_f1"]
    summary: dict[str, object] = {"fold_count": len(fold_metrics)}
    for metric_name in metric_names:
        values = [float(fold[metric_name]) for fold in fold_metrics]
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def run_temporal_backtest(
    spec: TrainableModelSpec,
    df: pd.DataFrame,
    target_column: str,
    labels: list[str],
    *,
    binary_task: bool,
    selected_features: list[str] | None = None,
) -> list[dict[str, object]]:
    X = df[selected_features or MODEL_FEATURES]
    y = df[target_column]
    fold_rows: list[dict[str, object]] = []

    for fold_number, (train_idx, test_idx) in enumerate(rolling_origin_splits(y), start=1):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < min(2, y.nunique()):
            continue
        estimator = spec.estimator_factory() if selected_features is None else _make_classifier_pipeline(
            clone(spec.estimator_factory().named_steps["classifier"]),
            selected_features,
        )
        estimator.fit(X.iloc[train_idx], y_train)
        if binary_task:
            probabilities = estimator.predict_proba(X.iloc[test_idx])
            positive_index = list(estimator.classes_).index("Permanent")
            threshold, y_pred = choose_binary_threshold(y.iloc[test_idx], probabilities[:, positive_index])
        else:
            threshold = None
            y_pred = estimator.predict(X.iloc[test_idx])
        metrics = compute_metrics(y.iloc[test_idx], pd.Series(y_pred, index=y.iloc[test_idx].index), labels=labels)
        metrics["fold"] = fold_number
        metrics["test_size"] = int(len(test_idx))
        metrics["train_size"] = int(len(train_idx))
        metrics["test_start"] = str(df.iloc[test_idx]["event_date"].min().date())
        metrics["test_end"] = str(df.iloc[test_idx]["event_date"].max().date())
        if threshold is not None:
            metrics["threshold"] = float(threshold)
        fold_rows.append(metrics)

    return fold_rows


def build_error_analysis(
    holdout_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object]]:
    error_df = holdout_df[
        [
            "fault_id",
            "fault_no",
            "event_date",
            "location_primary",
            "weather",
            "season",
            "substation_area",
            "row_count",
            "unique_apparatus_count",
            "comments_concat",
            "apparatus_concat",
        ]
    ].copy()
    error_df["actual_label"] = y_true.values
    error_df["predicted_label"] = y_pred.values
    error_df["is_error"] = error_df["actual_label"] != error_df["predicted_label"]

    summary = {
        "total_holdout_events": int(len(error_df)),
        "misclassified_events": int(error_df["is_error"].sum()),
        "error_rate": float(error_df["is_error"].mean()),
        "errors_by_weather": error_df.groupby("weather")["is_error"].sum().sort_values(ascending=False).to_dict(),
        "errors_by_season": error_df.groupby("season")["is_error"].sum().sort_values(ascending=False).to_dict(),
        "errors_by_area": error_df.groupby("substation_area")["is_error"].sum().sort_values(ascending=False).to_dict(),
    }
    return error_df, summary


def evaluate_feature_ablations(
    base_classifier,
    train_val_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_column: str,
    labels: list[str],
    *,
    binary_task: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    y_train = train_val_df[target_column]
    y_holdout = holdout_df[target_column]

    for feature_set_name, feature_list in FEATURE_SET_REGISTRY.items():
        model = _make_classifier_pipeline(clone(base_classifier), feature_list)
        model.fit(train_val_df[feature_list], y_train)
        if binary_task:
            probabilities = model.predict_proba(holdout_df[feature_list])
            positive_index = list(model.classes_).index("Permanent")
            threshold, predictions = choose_binary_threshold(y_holdout, probabilities[:, positive_index])
            metrics = compute_metrics(y_holdout, pd.Series(predictions, index=y_holdout.index), labels)
            metrics["threshold"] = float(threshold)
        else:
            predictions = pd.Series(model.predict(holdout_df[feature_list]), index=y_holdout.index)
            metrics = compute_metrics(y_holdout, predictions, labels)
        metrics["feature_set"] = feature_set_name
        metrics["feature_count"] = len(feature_list)
        results.append(metrics)

    return sorted(results, key=lambda item: item["macro_f1"], reverse=True)


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
        evaluation_summary["comparisons"][model_name]["confidence_intervals"] = bootstrap_metric_confidence_intervals(
            y_holdout,
            predictions,
            labels,
        )

    selected_spec = next(spec for spec in stage1_model_specs() if spec.name == selected_model_name)
    temporal_backtest = run_temporal_backtest(
        selected_spec,
        model_df,
        target_column="stage1_binary_label",
        labels=labels,
        binary_task=True,
    )
    evaluation_summary["temporal_backtest"] = {
        "folds": temporal_backtest,
        "summary": summarize_temporal_backtest(temporal_backtest),
    }

    ablation_results = evaluate_feature_ablations(
        selected_model.named_steps["classifier"],
        train_val_df,
        holdout_df,
        target_column="stage1_binary_label",
        labels=labels,
        binary_task=True,
    )
    evaluation_summary["feature_ablations"] = ablation_results

    error_analysis_df, error_analysis_summary = build_error_analysis(
        holdout_df,
        y_holdout,
        holdout_predictions[selected_model_name],
    )
    evaluation_summary["error_analysis"] = error_analysis_summary

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
    ablations_path = paths.stage1 / "stage1_feature_ablations.csv"
    error_analysis_path = paths.stage1 / "stage1_error_analysis.csv"
    confusion_path = paths.figures / "stage1_confusion_matrix.png"
    shap_note_path = paths.stage1 / "stage1_shap_note.txt"

    metrics_path.write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False)
    joblib.dump(selected_model, model_path)
    save_permutation_importance(selected_model, holdout_df[MODEL_FEATURES], y_holdout, importance_path)
    pd.DataFrame(ablation_results).to_csv(ablations_path, index=False)
    error_analysis_df.to_csv(error_analysis_path, index=False)
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
        "ablations": ablations_path,
        "error_analysis": error_analysis_path,
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
        evaluation_summary["comparisons"][model_name]["confidence_intervals"] = bootstrap_metric_confidence_intervals(
            y_holdout,
            predictions,
            labels,
        )

    selected_spec = next(spec for spec in stage2_model_specs() if spec.name == selected_model_name)
    temporal_backtest = run_temporal_backtest(
        selected_spec,
        merged,
        target_column="stage2_label",
        labels=labels,
        binary_task=False,
    )
    evaluation_summary["temporal_backtest"] = {
        "folds": temporal_backtest,
        "summary": summarize_temporal_backtest(temporal_backtest),
    }

    ablation_results = evaluate_feature_ablations(
        selected_model.named_steps["classifier"],
        train_val_df,
        holdout_df,
        target_column="stage2_label",
        labels=labels,
        binary_task=False,
    )
    evaluation_summary["feature_ablations"] = ablation_results

    error_analysis_df, error_analysis_summary = build_error_analysis(
        holdout_df,
        y_holdout,
        holdout_predictions[selected_model_name],
    )
    evaluation_summary["error_analysis"] = error_analysis_summary

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
    ablations_path = paths.stage2 / "stage2_feature_ablations.csv"
    error_analysis_path = paths.stage2 / "stage2_error_analysis.csv"
    confusion_path = paths.figures / "stage2_confusion_matrix.png"
    shap_note_path = paths.stage2 / "stage2_shap_note.txt"

    metrics_path.write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False)
    joblib.dump(selected_model, model_path)
    save_permutation_importance(selected_model, holdout_df[MODEL_FEATURES], y_holdout, importance_path)
    pd.DataFrame(ablation_results).to_csv(ablations_path, index=False)
    error_analysis_df.to_csv(error_analysis_path, index=False)
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
        "ablations": ablations_path,
        "error_analysis": error_analysis_path,
        "confusion_matrix": confusion_path,
    }
