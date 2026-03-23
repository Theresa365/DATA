# SCADA Fault Classification Report

## Data Summary
- Distribution workbook rows: 166
- System workbook rows: 618
- Distribution event records: 105
- System event records: 368
- Chronology anomalies quarantined from modeling: 2

## Stage 1 Benchmark
- Target: `Permanent` vs `Non-permanent` at event level.
- Train/validation events: 82
- Holdout events: 21
- Selected model: random-forest
- Selected threshold: 0.55
- Holdout macro F1: 0.914
- Holdout weighted F1: 0.950
- Holdout macro F1 95% CI: 0.488 to 1.000
- Temporal backtest mean macro F1: 0.956

## Stage 2 Fault-Family Study
- Taxonomy: `ground-related`, `phase-to-phase`, `three-phase`, `transformer/internal`, `operational-other`, `unknown/unclassifiable`.
- Curated distribution events: 105
- Unknown/unclassifiable events retained for QA only: 5
- Train/validation events: 78
- Holdout events: 20
- Selected model: logistic-regression
- Holdout macro F1: 1.000
- Holdout weighted F1: 1.000
- Holdout macro F1 95% CI: 1.000 to 1.000
- Temporal backtest mean macro F1: 0.410

## Stage 2 Rare-Class Merge
- Applied mapping: `{'three-phase': 'operational-other', 'phase-to-phase': 'operational-other'}`

## Evaluation Upgrades
- Time-aware rolling-origin backtests are included in both stage metric files.
- Bootstrap confidence intervals are reported for model holdout metrics.
- Feature ablation and misclassification exports are written under `outputs/stage1/` and `outputs/stage2/`.

## Domain Validation Pack
- Expert review templates are written under `outputs/validation/` for event aggregation and stage-2 labels.

## Notes
- Clean distribution modeling window spans 2023-01-06 to 2025-12-04.
- Results should be treated as exploratory because the event sample is small and utility-specific.
