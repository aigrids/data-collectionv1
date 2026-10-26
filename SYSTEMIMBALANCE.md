# SystemImbalance

Benchmarking the SystemImbalance task: dataset characterization and a
reference model (gradient boosted trees, from Kruse et al. [2]). Raw data
is published by Bundesnetzagentur | SMARD.de [1] under CC BY 4.0. The data
used here is available on Hugging Face: the raw series as downloaded from
SMARD in October 2026 at
https://huggingface.co/datasets/nikakhosravi-acs/SystemImbalance_raw, and
the standardized data at
https://huggingface.co/datasets/nikakhosravi-acs/SystemImbalance.

## New scripts and how to run them

All commands run from the repo root, with `config.yml` pointing at the
local dataset/results paths.

| Script | Purpose | Usage | Output |
|---|---|---|---|
| `processing/systemimbalance_process.py` | Merge the raw series into the standardized data | download and unpack `raw.tar.gz` from `SystemImbalance_raw`, set `path_root_source` and `path_root_target`, then `python processing/systemimbalance_process.py` | `systemimbalance_zones.csv`, `systemimbalance_national.csv` |
| `scripts/advanced_metrics_systemimbalance.py` | Basic and advanced characterization | `python scripts/advanced_metrics_systemimbalance.py <subtask>` | `results/analysis/advanced_metrics_SystemImbalance_<subtask>.json` |
| `scripts/benchmark_systemimbalance.py` | Reference model: gradient boosted trees | `python scripts/benchmark_systemimbalance.py <subtask> [data_frac] [n_search]` | `results/benchmark/benchmark_results_SystemImbalance_<subtask>.json` |

`subtask`: one of `odd_time_volume_predict1h`, `odd_space_volume_predict1h`,
`odd_spacetime_volume_predict1h`, `odd_time_price_predict1h`.
`src/scores.py` and `src/systemimbalance_features.py` are shared helpers,
not run directly. Output paths are where results get saved after running,
not existing files. The task is registered in `scripts/analyse.py`, but
`src/count.py` only supports WindFarm so far, so the basic counts below come
from the advanced characterization script.

## 1. Dataset characterization

Quarter-hourly series from SMARD [1] for the four German control zones
(50Hertz, Amprion, TenneT, TransnetBW), 22 June 2022 to 30 June 2026.
Missing values are below 0.5% in every series used. Missing and extreme
values are kept.

One data point is one zone and one predicted hour. The labels are the 4
quarter-hourly values of the signed system imbalance (volume at undersupply
plus volume at oversupply) in that hour. The features are the day-ahead
series of that hour (4 values each: forecasts of load, total generation,
wind onshore, wind offshore and solar, and the day-ahead price), the
system imbalance of the zone, and the summed system imbalance of the other
three zones, each for 24 hours (96 values). Windows move forward by one
hour, so labels do not overlap. For the price sub-task, the label is the
balancing energy price, and the history is the system imbalance of Germany
(sum of the four zones) and the balancing energy price.

**Publication delays.** Every history ends at the time at which it is
published. The transmission system operators publish the operational
system imbalance of each control zone 15 minutes after the end of each
quarter hour, and an estimate of the balancing energy price 30 minutes after
it [7]. For a predicted hour starting at 01:00, the imbalance history
therefore ends at 00:45 and the price history at 00:30. Final balancing
energy prices are published up to 20 working days after a month; the final
price stands in for the estimate.

**Horizon and inputs compared with the literature.** Forecasts of the
system imbalance are made from 15 minutes to 2 hours ahead, from recent
imbalance and day-ahead information: Salem et al. forecast the Norwegian
imbalance up to 2 hours ahead from the imbalance of the past 2 hours [8],
Van Gompel et al. the Belgian imbalance of the current and next two quarter
hours [9]. Narajewski forecasts the German balancing energy price 30
minutes before delivery and, like here, uses the imbalance only as far as it
is published, the newest value being the quarter hour that starts 60
minutes before delivery [10]. Kruse et al., whose model is used here,
predict from day-ahead information only [2].

| Sub-task | # data points | Train/Val/Test split | dim features (raw) | dim labels |
|---|---|---|---|---|
| odd_time_volume_predict1h | 141,020 | 50.0% / 10.0% / 40.0% | 216 | 4 |
| odd_space_volume_predict1h | 141,020 | 62.5% / 12.5% / 25.0% | 216 | 4 |
| odd_spacetime_volume_predict1h | 77,561 | 68.2% / 13.6% / 18.2% | 216 | 4 |
| odd_time_price_predict1h | 35,255 | 50.0% / 10.0% / 40.0% | 220 | 4 |

The three volume sub-tasks share the same four zones; `odd_space`/
`odd_spacetime` test on TransnetBW, which is not in the training data,
`odd_time` splits *when* (test from 20 Nov 2024). With only four zones,
`odd_space` validates on the last sixth of the time stamps of the three
training zones. The summed imbalance of the other zones has the same
meaning for every zone, including the unseen one.

**Why one hour.** The system imbalance depends strongly on its own recent
values and hardly on the day before: across the four zones its
autocorrelation is 0.86-0.91 after 15 minutes, 0.54-0.70 after one hour and
0.16-0.30 after one day. Predicting from day-ahead information alone is no
better than the training mean.

## 2. Reference model: gradient boosted trees [2]

Kruse et al. predict the activation of secondary control reserves in
Germany from day-ahead information with gradient boosted trees (LightGBM),
and publish their code under an MIT license [2]. Their model, day-ahead
feature recipe and hyper-parameter grid are used here for a different
target and different splits, and extended with the published history of
the system imbalance as input.

**Features**, per predicted quarter hour: the 96 historic values of the
system imbalance of the zone and of the other three zones (for the price
sub-task: of the system imbalance of Germany and of the balancing energy
price); the day-ahead forecasts of load, total generation, wind onshore,
wind offshore and solar, and the day-ahead price; the change of each with
respect to the previous quarter hour; hour of the day, day of the week and
position in the predicted hour (207 features; 209 for the price sub-task,
which also has scheduled net export).

**Model:** `LGBMRegressor` with the fixed settings of [2], once with squared
error and once with the 90% quantile loss. Hyper-parameters are selected
from the grid of [2] (288 combinations) by early stopping on the validation
split, not by random cross-validation as in [2], since neighbouring quarter
hours in training and validation folds would leak. Naive predictors for
comparison: training mean, mean daily profile, and persistence (repeating
the last published value of the label).

**Check of the ported code:** refitting the models of [2] with their
published hyper-parameters on their published data reproduces their test
scores (R2 of 0.441 against 0.446 and 0.474 against 0.476 for the day-ahead
model).

### 2.1 Point-forecast accuracy (all 4 sub-tasks)

Run on `claix` (partition `c23ms`, Intel Xeon Platinum 8468, 8 cores and
16GB RAM per sub-task; Python 3.12.3, LightGBM 4.7.0, scikit-learn 1.9.1),
with the full grid of 288 hyper-parameter combinations. All numbers are on
the test split, over the four predicted quarter hours. Units are MWh for the
volume sub-tasks and EUR/MWh for the price sub-task.

Gradient boosted trees (GBT):

| Sub-task | MAE | RMSE | R2 | Sign accuracy |
|---|---|---|---|---|
| odd_time_volume_predict1h | 54.2 | 76.5 | 0.526 | 0.764 |
| odd_space_volume_predict1h | 26.0 | 36.6 | 0.531 | 0.769 |
| odd_spacetime_volume_predict1h | 28.4 | 40.1 | 0.535 | 0.772 |
| odd_time_price_predict1h | 52.3 | 132.8 | 0.264 | 0.899 |

Persistence (last published value of the label, repeated for all four
quarter hours):

| Sub-task | MAE | RMSE | R2 | Sign accuracy |
|---|---|---|---|---|
| odd_time_volume_predict1h | 61.1 | 86.7 | 0.392 | 0.751 |
| odd_space_volume_predict1h | 27.5 | 39.4 | 0.459 | 0.771 |
| odd_spacetime_volume_predict1h | 30.2 | 43.4 | 0.457 | 0.771 |
| odd_time_price_predict1h | 61.6 | 166.7 | -0.160 | 0.897 |

Training mean:

| Sub-task | MAE | RMSE | R2 |
|---|---|---|---|
| odd_time_volume_predict1h | 80.6 | 111.2 | 0.000 |
| odd_space_volume_predict1h | 40.1 | 55.0 | -0.055 |
| odd_spacetime_volume_predict1h | 44.5 | 60.8 | -0.068 |
| odd_time_price_predict1h | 65.9 | 156.2 | -0.018 |

The mean daily profile performs like the training mean in all sub-tasks and
is not listed. For the price, sign accuracy is the share of correctly
predicted signs of the price itself.

Improvement of the gradient boosted trees over persistence:

| Sub-task | MAE reduction | RMSE reduction | R2 (GBT - persistence) |
|---|---|---|---|
| odd_time_volume_predict1h | 11% | 12% | +0.135 |
| odd_space_volume_predict1h | 5% | 7% | +0.073 |
| odd_spacetime_volume_predict1h | 6% | 8% | +0.079 |
| odd_time_price_predict1h | 15% | 20% | +0.424 |

Accuracy falls with the lead time:

| Sub-task | R2 at +15 min | +30 min | +45 min | +60 min |
|---|---|---|---|---|
| odd_time_volume_predict1h | 0.625 | 0.555 | 0.493 | 0.431 |
| odd_space_volume_predict1h | 0.664 | 0.567 | 0.479 | 0.412 |
| odd_spacetime_volume_predict1h | 0.668 | 0.569 | 0.485 | 0.414 |
| odd_time_price_predict1h | 0.290 | 0.282 | 0.259 | 0.222 |

The balancing energy price is harder to predict than the system imbalance.
It is set by the imbalance and by the prices of the balancing energy
activated in the delivery quarter hour itself, and it has rare extreme
values that dominate RMSE and R2; MAE and the improvement over persistence
are the more informative numbers for this sub-task. Persistence is worse
than the training mean here, because the newest published price is 30 to 45
minutes older than the predicted quarter hours.

90% quantile model (quantile loss; share of labels below the prediction):

| Sub-task | Quantile loss (GBT / constant) | Coverage (GBT / constant) |
|---|---|---|
| odd_time_volume_predict1h | 12.63 / 20.24 | 0.880 / 0.871 |
| odd_space_volume_predict1h | 6.82 / 12.72 | 0.962 / 0.976 |
| odd_spacetime_volume_predict1h | 7.34 / 12.55 | 0.953 / 0.960 |
| odd_time_price_predict1h | 14.15 / 28.36 | 0.921 / 0.987 |

"constant" is the 90% quantile of the training labels.

Selected hyper-parameters and general performance metrics of the squared
error model:

| Sub-task | Trees / leaves / learning rate | Train time (sec) | Inference per data point (ms) | Batch scaling factor (b = 4 / 16 / 64) |
|---|---|---|---|---|
| odd_time_volume_predict1h | 765 / 900 / 0.01 | 26.2 | 0.46 | 0.32 / 0.16 / 0.12 |
| odd_space_volume_predict1h | 997 / 900 / 0.01 | 41.5 | 0.64 | 0.41 / 0.29 / 0.26 |
| odd_spacetime_volume_predict1h | 560 / 900 / 0.01 | 17.9 | 0.53 | 0.30 / 0.14 / 0.10 |
| odd_time_price_predict1h | 51 / 10 / 0.1 | 0.6 | 0.32 | 0.26 / 0.07 / 0.02 |

| Sub-task | Perturbation robustness (sigma = 0.01 / 0.05 / 0.1) | Test RMSE with 5% / 10% / 25% / 50% / 100% of training data |
|---|---|---|
| odd_time_volume_predict1h | 286.1 / 183.5 / 157.3 | 80.0 / 79.0 / 77.8 / 76.9 / 76.6 |
| odd_space_volume_predict1h | 252.6 / 156.7 / 145.7 | 39.6 / 37.7 / 37.3 / 36.9 / 36.6 |
| odd_spacetime_volume_predict1h | 251.6 / 160.4 / 149.3 | 41.8 / 41.2 / 40.8 / 40.5 / 40.1 |
| odd_time_price_predict1h | 463.4 / 336.3 / 275.2 | 140.3 / 138.2 / 140.3 / 135.8 / 135.4 |

Train time is one fit of the selected model. Perturbation robustness adds
Gaussian noise of sigma times the training standard deviation to every
feature except the calendar features, and is the mean L2 distance between
the predictions of a data point with and without noise, divided by sigma.

Whole jobs, including the hyper-parameter search for both models (SLURM
accounting):

| Sub-task | Job time | Energy (kWh)* | Peak memory (GB) |
|---|---|---|---|
| odd_time_volume_predict1h | 2:22:56 | 1.42 | 3.5 |
| odd_space_volume_predict1h | 3:06:24 | 2.01 | 3.7 |
| odd_spacetime_volume_predict1h | 2:05:20 | 1.41 | 2.6 |
| odd_time_price_predict1h | 0:31:05 | 0.37 | 1.7 |

*As reported by SLURM for the job (`ConsumedEnergy`); how the cluster
attributes node energy to a job that shares a node has not been checked.

### 2.2 Advanced characterization

Two groups of quantities: model size thresholds and dataset scores as
defined by Aryandoust et al. [3], and the train-test distribution shift
used for WindFarm.

**Notation used below:**

| Symbol | Meaning |
|---|---|
| $n$, $s_{tr}$ | number of data points; share used for training |
| $D_x$, $D_y$ | dimension of features; dimension of labels |
| $ipt$, $sft$ | interpolation threshold; smooth function threshold |
| JSD | Jensen-Shannon divergence (base 2, between 0 and 1) |
| $d$, $\sigma_{pooled}$ | train-test distribution-shift statistic; its pooled standard deviation |

| Quantity | Formula | Status |
|---|---|---|
| Interpolation threshold | $ipt=s_{tr}\,n\,D_y$ | Definition from [3], based on [4] |
| Smooth function threshold | $sft=ipt\cdot D_x$ | Definition from [3], based on [5] |
| SImb-score | mean over columns of JSD between the column's histogram ($10^4$ bins, Tukey outliers removed) and a uniform distribution | From [3], numpy port of their code |
| STood-score | mean over columns of JSD between the column's histograms in training and testing data | From [3], numpy port of their code |
| IO-score | mean over feature-label pairs of $\tfrac{2}{\pi}\arctan\lvert\Delta l/\Delta f\rvert$ between consecutive points sorted by the feature | From [3], numpy port of their code |
| Outlier-score | mean severity of values beyond the inner Tukey fences, 0 at $1.5\,$IQR to 1 at $3\,$IQR | From [3], numpy port of their code |
| Train-test distribution shift $\lvert d\rvert$ | $d=\dfrac{\bar y_{train}-\bar y_{test}}{\sigma_{pooled}}$, $\sigma_{pooled}=\sqrt{(\sigma^2_{train}+\sigma^2_{test})/2}$, on the mean label of a data point | Formula from [6], as for WindFarm |

The scores follow the order of steps of the code of [3]: standardize
features and labels with training statistics, IO-score on all data, PCA
keeping 99% of the variance of the features, then SImb-, STood- and
Outlier-score on the PCA components. Scores need complete data: wind
offshore forecasts of zones without wind offshore generation are set to
zero, and the few data points with other missing values are left out (at
most 0.8% of a split).

**Results, all 4 sub-tasks** (computed on a laptop; the quantities do not
depend on hardware):

| Sub-task | $ipt$ | $sft$ | PCA components |
|---|---|---|---|
| odd_time_volume_predict1h | 282,032 | 60,918,912 | 168 |
| odd_space_volume_predict1h | 352,548 | 76,150,368 | 165 |
| odd_spacetime_volume_predict1h | 211,523 | 45,689,183 | 167 |
| odd_time_price_predict1h | 70,508 | 15,511,760 | 180 |

"random" is the STood-score of the same data split at random into parts of
the same sizes (mean of three splits), that is, the value the score takes
without any distribution shift.

| Sub-task | SImb train / test | STood test (random) | STood test, labels (random) | IO | Outlier | Distribution shift $\lvert d\rvert$ |
|---|---|---|---|---|---|---|
| odd_time_volume_predict1h | 0.145 / 0.153 | 0.067 (0.064) | 0.110 (0.067) | 0.999 | 0.522 | 0.007 |
| odd_space_volume_predict1h | 0.141 / 0.182 | 0.114 (0.081) | 0.480 (0.083) | 0.999 | 0.518 | 0.150 |
| odd_spacetime_volume_predict1h | 0.159 / 0.272 | 0.192 (0.176) | 0.515 (0.149) | 0.998 | 0.520 | 0.185 |
| odd_time_price_predict1h | 0.248 / 0.273 | 0.230 (0.234) | 0.254 (0.235) | 0.996 | 0.598 | 0.081 |

With tens of thousands of data points per split, the STood-score separates
the splits: it is at the level of a random split when the test set is a
later period, and higher when the test set is the unseen zone, much more so
for the labels than for the features, since the features include the
imbalance of the other zones, which looks alike for every zone. The
distribution shift $\lvert d\rvert$ shows the same pattern. The
random-split value is still needed to read the score, because it depends on
the number of data points: two samples of 3,000 points from the same normal
distribution score 0.66, two samples of 200,000 points score 0.02. The
SImb-score likewise falls with the number of data points. The IO-score is
at its maximum everywhere (on synthetic data: 0.80 for an exact linear
relation without noise, 1.00 for pure noise), and the Outlier-score is
close to the 0.52 of a normal distribution.

Aleatoric and epistemic uncertainty from an ensemble of probabilistic
models, as computed for WindFarm, is not available yet.

## References

[1] Bundesnetzagentur. SMARD.de - Strommarktdaten. https://www.smard.de

[2] Kruse, J., Schäfer, B. & Witthaut, D. (2022). Secondary control
activation analysed and predicted with explainable AI. *Electric Power
Systems Research*, 212, 108489. Code:
https://github.com/johkruse/XAI-for-aFRR-activation

[3] Aryandoust, A., Rigoni, T., Di Stefano, F. & Patt, A. (2023). Unified
machine learning tasks and datasets for enhancing renewable energy.
arXiv:2311.06876. Code:
https://github.com/ArsamAryandoust/EnergyTransitionTasks

[4] Belkin, M., Hsu, D., Ma, S. & Mandal, S. (2019). Reconciling modern
machine-learning practice and the classical bias-variance trade-off.
*PNAS*, 116(32), 15849-15854.

[5] Bubeck, S. & Sellke, M. (2021). A Universal Law of Robustness via
Isoperimetry. *NeurIPS*.

[6] Austin, P. C. (2009). Balance diagnostics for comparing the
distribution of baseline covariates between treatment groups in
propensity-score matched samples. *Statistics in Medicine*, 28(25),
3083-3107.

[7] 50Hertz, Amprion, TenneT & TransnetBW (2023). Erläuterungen zu den
Veröffentlichungen auf der Informationsplattform, sections 2.4 and 3.1.
https://www.netztransparenz.de

[8] Salem, T. S., Kathuria, K., Ramampiaro, H. & Langseth, H. (2019).
Forecasting intra-hour imbalances in electric power systems. *Proceedings
of the AAAI Conference on Artificial Intelligence*, 33, 9595-9600.

[9] Van Gompel, J., Claessens, B. & Develder, C. (2025). Probabilistic
forecasting of power system imbalance using neural network-based
ensembles. *Applied Energy*. arXiv:2404.14836.

[10] Narajewski, M. (2022). Probabilistic forecasting of German electricity
imbalance prices. *Energies*, 15(14), 4976.
