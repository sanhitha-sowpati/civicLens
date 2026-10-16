# Evaluation report - Where Should We Fix First?

Model version: `20261016104556-6c5a2cff56`  |  Trained (UTC): 2026-10-16T10:45:56.673540+00:00


## 1. What this model is (and is not)

* The supplied data contains **no genuine target** - no municipal decision, repair-urgency or
  official priority field. (`priority_reported` in the complaints file is a citizen-chosen flag,
  not an official decision.)
* The model is therefore trained on a **documented proxy priority score** (0-100) built from
  incident severity, injuries, unresolved incidents, complaints, open complaints and static
  location conditions (formula in the README and `preprocessing_config.json`).
* **This model does not learn real municipal priorities.** It forecasts the proxy score of the
  next 60 days from records available up to the
  prediction time.
* The data is synthetic (see `data/DATASET_README.txt`); results must not be used for real decisions.

## 2. Evaluation design

Point-in-time snapshots every 15 days. Features use only
records up to the cutoff (unresolved/open status reconstructed *at* the cutoff); the label is the
proxy score of the following 60 days. Splits are
time-ordered with a purge gap so label windows never overlap across splits:

- train: 2025-01-31 to 2025-04-16 (6 cutoffs)
- val: 2025-06-15 to 2025-07-15 (3 cutoffs)
- test: 2025-09-13 to 2025-10-28 (4 cutoffs)

The test split was used only for the numbers below - not for label construction, tuning,
early stopping, or the priority-level thresholds.

## 3. Metrics

| Split | Model | MAE \| RMSE \| R² |
|---|---|---|
| train | CatBoost | 11.36 | 13.82 | 0.108 |
| train | Train-mean baseline | 12.01 | 14.63 | 0.000 |
| train | Persistence (prev. 60d proxy) | 16.03 | 20.05 | -0.878 |
| validation | CatBoost | 11.44 | 13.88 | 0.060 |
| validation | Train-mean baseline | 11.72 | 14.32 | -0.000 |
| validation | Persistence (prev. 60d proxy) | 16.38 | 20.40 | -1.032 |
| test | CatBoost | 11.79 | 14.30 | 0.036 |
| test | Train-mean baseline | 11.93 | 14.56 | -0.000 |
| test | Persistence (prev. 60d proxy) | 15.89 | 19.81 | -0.850 |

Rank-oriented metrics (per cutoff, averaged): test Spearman = 0.184
(persistence 0.065); validation Spearman =
0.229.

### Honest reading

* On the held-out test period the model shows **weak predictive skill** (R² = 0.036).
* Its MAE is 11.79 versus 11.93 for the train-mean baseline (+1.2% MAE improvement; negative = worse) and 15.89 for the persistence baseline (previous 60-day proxy score).
* Top-5% overlap with the truly highest-scoring locations: model 10.5% (chance level 5%), persistence 12.0%.

Context: the label's static location-condition component alone has a standard deviation of
3.4 points, versus 14.6
for the full label. Event-driven parts of the label are largely random from one window to the
next in this dataset, which caps how well *any* model can forecast it. Do not read the model's
scores as stronger evidence than these metrics support.

## 4. Plots

* `actual_vs_predicted.png` - scatter and decile calibration on the test split
* `feature_importance.png` / `feature_importance.csv`
* `score_distribution.png`

Top features:

- `road_condition_score`: 27.63
- `pedestrian_activity_score`: 17.42
- `population_density_band`: 11.95
- `near_school`: 6.68
- `zone`: 6.46
- `near_hospital`: 5.58
- `road_type`: 5.30
- `high_priority_complaint_rate_per_30d`: 2.52
- `unresolved_incident_share`: 2.15
- `mean_severity_score`: 1.87

## 5. Priority levels

Thresholds are quantiles of **predicted validation scores** (fixed at training time, so levels
do not shuffle when a new report arrives): Medium >= 30.22 (50th pct),
High >= 32.57 (80th pct), Critical >= 35.42 (95th pct),
otherwise Low. Levels are therefore *relative* triage bands. Current distribution: {'Critical': 57, 'High': 150, 'Medium': 297, 'Low': 496}.
Displayed scores are clipped to 0-100.

## 6. Top-ranked locations (as of 2025-12-31 23:50:00)

|   rank | location_id   |   predicted_priority_score | priority_level   |   incident_count |   complaint_count |   unresolved_incident_count | road_condition   |
|-------:|:--------------|---------------------------:|:-----------------|-----------------:|------------------:|----------------------------:|:-----------------|
|      1 | L0556         |                      40.47 | Critical         |               14 |                 9 |                           6 | Critical         |
|      2 | L0538         |                      39.94 | Critical         |                6 |                 8 |                           3 | Critical         |
|      3 | L0985         |                      39.75 | Critical         |               10 |                 9 |                           4 | Poor             |
|      4 | L0919         |                      39.6  | Critical         |               12 |                 7 |                           2 | Poor             |
|      5 | L0332         |                      39.25 | Critical         |               14 |                11 |                           5 | Poor             |
|      6 | L0467         |                      39.25 | Critical         |               10 |                 5 |                           2 | Critical         |
|      7 | L0240         |                      39.17 | Critical         |                6 |                 3 |                           1 | Critical         |
|      8 | L0860         |                      39.11 | Critical         |                8 |                 6 |                           1 | Poor             |
|      9 | L0072         |                      38.94 | Critical         |               13 |                10 |                           6 | Critical         |
|     10 | L0343         |                      38.57 | Critical         |               14 |                 5 |                           3 | Critical         |
|     11 | L0484         |                      38.09 | Critical         |               12 |                 4 |                           4 | Critical         |
|     12 | L0073         |                      37.93 | Critical         |               12 |                14 |                           5 | Critical         |
|     13 | L0826         |                      37.82 | Critical         |               11 |                 7 |                           4 | Critical         |
|     14 | L0214         |                      37.78 | Critical         |               11 |                10 |                           3 | Poor             |
|     15 | L0688         |                      37.67 | Critical         |               19 |                 6 |                           7 | Critical         |

Full table: `ranked_locations.csv`. `supporting_evidence` is computed from the underlying
records; `recent_proxy_score_60d` is the plain formula applied to the last 60 days of records.

## 7. Limitations

* Labels are a proxy defined by the project authors, not ground truth. Weights and saturation
  constants are judgement calls (see README) and could reasonably differ.
* Synthetic data: incident counts show essentially no persistence per location, so forecasting
  skill beyond static conditions is expected to be low.
* About one year of history limits time-aware evaluation (few cutoffs, correlated windows).
* Rates are normalised per 30 days of history so snapshots at different dates are comparable;
  behaviour on much longer histories than seen in training is untested.
* Latitude/longitude, location IDs and exact population density are deliberately excluded to
  prevent location-identity shortcuts; spatial effects are therefore not modelled.
* The model is not retrained when reports arrive; retrain explicitly with `scripts/train.py`.
