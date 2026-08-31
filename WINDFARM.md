# WindFarm

Benchmarking the WindFarm task: dataset characterization, a reference
model (DMST, from the KDD Cup 2022 winning solution [1]), and an
optional simple baseline (ridge regression). Raw data is SDWPF [2].

## New scripts and how to run them

All commands run from the repo root, with `config.yml` pointing at the
local dataset/results paths.

| Script | Purpose | Usage | Output |
|---|---|---|---|
| `scripts/dataset_stats_windfarm.py` | Raw-data stats | `python scripts/dataset_stats_windfarm.py` | `results/windfarm_dataset_stats.json` |
| `scripts/dmst_windfarm.py` | Reference model: DMST, point-forecast accuracy | `python scripts/dmst_windfarm.py <subtask> [data_frac] [n_epochs]` | `results/dmst/dmst_windfarm_<subtask>.json` |
| `scripts/probabilistic_dmst_windfarm.py` | Reference model: DMST-based aleatoric/epistemic/total uncertainty | `python scripts/probabilistic_dmst_windfarm.py <subtask> [data_frac] [n_epochs] [n_ensemble]` | `results/advanced_metrics/probabilistic_dmst_windfarm_<subtask>.json`, plus a checkpoint per ensemble member under `results/advanced_metrics/checkpoints/<subtask>/` |
| `scripts/advanced_metrics_windfarm.py` | Train-test distribution shift | `python scripts/advanced_metrics_windfarm.py <subtask> [data_frac]` | `results/advanced_metrics/advanced_metrics_windfarm_<subtask>.json` |
| `scripts/benchmark_windfarm.py` (optional baseline) | Ridge regression | `python scripts/benchmark_windfarm.py <subtask> [data_frac] [alpha]` | `results/benchmark/ridge_windfarm_<subtask>_alpha<alpha>.json` |

`subtask`: one of `odd_time_predict48h`, `odd_space_predict48h`,
`odd_spacetime_predict48h`, `odd_time_predict72h`, `odd_space_predict72h`,
`odd_spacetime_predict72h`. `src/windfarm_features.py` and
`src/windfarm_graph.py` are shared helpers, not run directly. Output
paths are where results get saved after running, not existing files.

## Bugs found and fixed

Found while first getting WindFarm running; fixed on this branch.

| # | Bug | Fix | Repo |
|---|---|---|---|
| 1 | Data loading windowed one timestep at a time instead of one day at a time - ~11M near-duplicate records, memory climbing past 90GB and still rising | Slide by a full day instead of one measurement (~78,000 records, ~12GB, stable) | `aigrids` |
| 2 | `data_frac` (subsampling) grabbed random individual rows *before* building day-long windows, breaking window contiguity | Build windows first, subsample finished windows after | `aigrids` |
| 3 | `analyse.py` pointed at a nonexistent `config_arsam.yml` | Use `config.yml` | `data-collectionv1` |
| 4 | `download.py` imported the old package name `aidotgrids` | Fixed to `aigrids` | `data-collectionv1` |
| 5 | Sensor readings included physically impossible values (temperature at exactly -273.17°C, wind/nacelle direction far outside ±360°) | Masked in the `aigrids` loader | `aigrids` |
| 6 | Leftover `aidotgrids` naming in 5 test files + 1 docstring, would crash on import | Fixed | `aigrids` |

## 1. Dataset characterization

134 turbines [2], ~11.4M measurements (10-min intervals, 2020-2022), ~19.5%
of expected timesteps entirely missing, 4.4% of sensor values missing
within present rows.

| Sub-task | # data points | Train/Val/Test split | dim features (raw) | dim labels |
|---|---|---|---|---|
| odd_time_predict48h | 77,854 | 50.3% / 9.6% / 40.1% | 2,307 | 288 |
| odd_space_predict48h | 78,524 | 50.0% / 9.7% / 40.3% | 2,307 | 288 |
| odd_spacetime_predict48h | 78,524 | 50.0% / 9.7% / 40.3% | 2,307 | 288 |
| odd_time_predict72h | 77,452 | 50.3% / 9.5% / 40.1% | 2,307 | 432 |
| odd_space_predict72h | 78,390 | 50.0% / 9.7% / 40.3% | 2,307 | 432 |
| odd_spacetime_predict72h | 78,390 | 50.0% / 9.7% / 40.3% | 2,307 | 432 |

All 6 sub-tasks share the same 134 turbines; `odd_space`/`odd_spacetime`
split *which* turbines go into train vs. test, `odd_time` splits *when*.

## 2. Reference model: DMST [1]

Team HIK won the KDD Cup 2022 wind power forecasting competition with
FDSTT, a two-part model combining a deep spatio-temporal network with a
tree-based model [1]. No public code exists for it, so DMST - the deep-learning half, and the part reimplemented here - was
built from the description in the paper.

The other half of FDSTT (a LightGBM tree model plus a hand-tuned
ensemble-gating rule) is intentionally out of scope: [1] calibrated that
rule to Baidu's own competition test windows, and it doesn't transfer
to the `odd_time`/`odd_space`/`odd_spacetime` splits used here - adding
a second model family is also a larger scope than this benchmark's
characterization goal calls for.

**Graph construction.** The temporal turbine data are combined with a
multi-relational graph describing spatial and semantic relationships
between turbines: the spatial graph links each turbine to its 5
nearest turbines by Euclidean distance; the semantic graph links each
turbine to the 5 turbines with the highest differential similarity in
wind-speed changes ($K{=}5$ for both, [1]). Since `aigrids` windows
each turbine's data in isolation, each record is matched back to its
turbine by position, and neighbors' raw wind-speed history is pulled
from the source SCADA data.

**Architecture** (hidden sizes and embedding dimension per [1]):
a GRU encoder (hidden size 64) reads the 144-step
history, each timestep concatenating a 5-dim turbine embedding, 4-dim
aggregated wind speed (own reading plus the spatial- and
semantic-neighbor means), and 10 other features (temperature,
wind/nacelle direction, pitch, reactive power, historic power,
hour-of-day, position). A `GRUCell` decoder (hidden size 32) then runs
autoregressively over the forecast horizon: each step takes the turbine
embedding, the previous step's power (teacher-forced during training),
hour-of-day, and position, and a `Linear(32,1)` readout produces the
prediction. Trained with Adam (lr 1e-3), MSE loss, gradient clipping at
5.0, 20 epochs, on `ghost` (Tesla P40/A2 GPUs).

### 2.1 Point-forecast accuracy (all 6 sub-tasks)

| Sub-task | Val MAE (DMST/mean) | Val RMSE (DMST/mean) | Test MAE (DMST/mean) | Test RMSE (DMST/mean) |
|---|---|---|---|---|
| odd_time_predict48h | 361.7 / 385.3 | 506.2 / 485.7 | 301.3 / 339.0 | 424.6 / 417.3 |
| odd_space_predict48h | 327.5 / 351.6 | 418.3 / 435.0 | 319.6 / 346.3 | 411.6 / 427.6 |
| odd_spacetime_predict48h | 313.4 / 341.4 | 398.2 / 418.1 | 314.4 / 347.3 | 412.2 / 429.7 |
| odd_time_predict72h | 379.3 / 393.6 | 512.0 / 498.4 | 311.3 / 335.6 | 418.9 / 415.1 |
| odd_space_predict72h | 327.0 / 348.5 | 429.2 / 431.4 | 316.1 / 341.2 | 415.8 / 421.5 |
| odd_spacetime_predict72h | 310.8 / 337.1 | 406.9 / 415.7 | 321.0 / 345.3 | 421.3 / 429.3 |

DMST [1] beats the mean baseline on MAE across every sub-task. On RMSE, it
beats the mean baseline on every `odd_space`/`odd_spacetime` sub-task but
is worse on both `odd_time` sub-tasks - the model has no explicit
calendar/seasonal awareness, so generalizing to a genuinely later time
period is harder for it than generalizing to unseen turbines.

Training/inference time, 20 epochs, on `ghost`:

| Sub-task | Train time (sec) | Infer time (sec) |
|---|---|---|
| odd_time_predict48h | 326.5 | 3.03 |
| odd_space_predict48h | 267.1 | 7.78 |
| odd_spacetime_predict48h | 231.6 | 3.37 |
| odd_time_predict72h | 383.0 | 2.87 |
| odd_space_predict72h | 283.5 | 4.05 |
| odd_spacetime_predict72h | 268.7 | 3.85 |

### 2.2 Advanced characterization

Two quantities, not to be confused: train-test distribution shift and a
model-based aleatoric/epistemic/total decomposition.

**Notation used below:**

| Symbol | Meaning |
|---|---|
| $m$, $M$ | ensemble member index; ensemble size ($M{=}5$) |
| $h$, $x$ | forecast horizon step; one test instance (input) |
| $\mu_{m,h}(x)$, $\sigma_{m,h}^2(x)$ | member $m$'s predicted mean/variance at step $h$ |
| $s_{m,h}(x)=\log\sigma_{m,h}^2(x)$ | network's log-variance output |
| $D$ | number of (record, horizon-step) pairs in a training batch |
| NLL | negative log-likelihood (the loss function used) |
| $U_{aleatoric}$, $U_{epistemic}$, $U_{total}$ | aleatoric / epistemic / total predictive variance |
| $d$, $\sigma_{pooled}$ | train-test distribution-shift statistic; its pooled standard deviation |
| OOD | out-of-distribution |

| Quantity | Formula | Status |
|---|---|---|
| Train-test distribution shift $\lvert d\rvert$ | $d=\dfrac{\bar y_{train}-\bar y_{test}}{\sigma_{pooled}}$, $\sigma_{pooled}=\sqrt{(\sigma^2_{train}+\sigma^2_{test})/2}$, on the per-record horizon-mean of `Patv`, population variance | Formula from [3], repurposed from treatment/control to train/test; we add the absolute value ([3]'s $d$ is signed) |
| $U_{aleatoric}$, $U_{epistemic}$, $U_{total}$ | see below | Concept from [4]; formula derived by us, following its citation of [5] for the ensemble |

Each member $m$ outputs, per horizon step $h$, $\mu_{m,h}(x)$ and
$s_{m,h}(x)=\log\sigma_{m,h}^2(x)$ (clamped to $[-10,10]$) of
$Y\mid X{=}x,\theta_m\sim\mathcal{N}(\mu_{m,h}(x),\sigma_{m,h}^2(x))$
[4], trained with Gaussian NLL (Adam, `weight_decay=1e-2`),
the same loss as [6]:

$$L=\frac{1}{D}\sum_i\Big[\tfrac{1}{2}e^{-s_i}(y_i-\mu_i)^2+\tfrac{1}{2}s_i\Big]$$

$M{=}5$ members share one architecture, each trained on the full
training set with only the random seed varying - not bootstrap, which
[5] found hurts neural-net ensembles. Per test instance $x$ and
horizon step $h$:

$$U_{aleatoric,h}(x)=\frac{1}{M}\sum_{m=1}^{M} \sigma_{m,h}^2(x), \quad U_{epistemic,h}(x)=\frac{1}{M}\sum_{m=1}^{M}\big(\mu_{m,h}(x)-\bar\mu_h(x)\big)^2, \quad U_{total,h}=U_{aleatoric,h}+U_{epistemic,h}$$

the standard decomposition for an equally-weighted Gaussian mixture,
dividing by $M$ not $M{-}1$. Computed per (instance,
horizon step), then averaged over instances, then over horizon steps -
matching how §2.1's MAE/RMSE are aggregated.

High $U_{aleatoric}$ reflects irreducible task variability;
high $U_{epistemic}$ reflects disagreement between independently-seeded
members, i.e. how well-determined the model is. Neither is a formal
OOD test, and calibration is not verified here.

**Results, all 6 sub-tasks** ($M{=}5$, 20 epochs, on `claix`, one
NVIDIA H100):

| Sub-task | Distribution shift $\lvert d\rvert$ | $U_{aleatoric}$ (kW²) | $U_{epistemic}$ (kW²) | $U_{total}$ (kW²) |
|---|---|---|---|---|
| odd_time_predict48h | 0.0485 | 2,802.9 | 6,524.2 | 9,327.1 |
| odd_space_predict48h | 0.0115 | 3,449.4 | 11,343.3 | 14,792.7 |
| odd_spacetime_predict48h | 0.0066 | 6,276.0 | 13,856.5 | 20,132.5 |
| odd_time_predict72h | 0.0416 | 2,564.8 | 4,490.6 | 7,055.4 |
| odd_space_predict72h | 0.0171 | 3,752.7 | 7,428.6 | 11,181.3 |
| odd_spacetime_predict72h | 0.0226 | 5,345.8 | 36,384.3 | 41,730.0 |

Ensemble training time, on `claix` (one H100 per job, 4 CPUs, 32GB RAM):

| Sub-task | Train time (sec) | CPU+DRAM energy (kWh)* |
|---|---|---|
| odd_time_predict48h | 1,097.2 | 0.173 |
| odd_space_predict48h | 999.0 | 0.170 |
| odd_spacetime_predict48h | 965.9 | 0.157 |
| odd_time_predict72h | 1,543.6 | 0.230 |
| odd_space_predict72h | 1,142.5 | 0.181 |
| odd_spacetime_predict72h | 1,129.0 | 0.172 |

*CPU+DRAM only (SLURM/RAPL accounting) - GPU power isn't tracked on
this cluster and isn't included; these numbers understate true energy
use.

## 3. Optional baseline: ridge regression

Not the reference model - a simple, closed-form baseline kept for
reference, since one reproducible baseline per task is required and
ridge is the simplest option that works after fixing its
numerical instability (see below).

Fits $W,b$ minimizing
$J(W,b)=\sum_i\lVert y_i-Wx_i-b\rVert^2+\alpha\lVert W\rVert_F^2$; raw
144-step sensor sequences are highly collinear, so each sensor's
sequence is collapsed to 3 numbers (last value, mean, trend) before
fitting - 2,307 raw features become 51.

| Sub-task | Val MAE (ridge/mean) | Val RMSE (ridge/mean) | Test MAE (ridge/mean) | Test RMSE (ridge/mean) |
|---|---|---|---|---|
| odd_time_predict48h | 418.86 / 385.32 | 516.82 / 485.69 | 328.73 / 338.98 | 421.64 / 417.28 |
| odd_space_predict48h | 305.38 / 344.60 | 394.45 / 428.86 | 304.29 / 344.61 | 392.84 / 428.17 |
| odd_spacetime_predict48h | 305.69 / 343.37 | 392.85 / 425.62 | 307.42 / 347.68 | 395.64 / 430.09 |
| odd_time_predict72h | 426.71 / 393.65 | 526.94 / 498.42 | 336.18 / 335.62 | 429.57 / 415.07 |
| odd_space_predict72h | 307.58 / 340.05 | 392.27 / 418.70 | 311.27 / 342.59 | 395.13 / 423.21 |
| odd_spacetime_predict72h | 307.65 / 340.35 | 394.28 / 421.43 | 309.48 / 342.42 | 396.67 / 424.40 |

(`alpha=100`.) Beats the mean baseline on every `odd_space`/
`odd_spacetime` sub-task, loses on both `odd_time` sub-tasks - it has no
notion of season/calendar at all, so it cannot generalize across time
the way it can across turbines.

## References

[1] Li, L., Sun, Q., Geng, D., Jian, C., Wu, D. & Pu, S. (2022).
Complementary Fusion of Deep Spatio-Temporal Network and Tree Model for
Wind Power Forecasting (Team:HIK). *KDD '22 Workshop*.

[2] Zhou, J., Lu, X., Xiao, Y., Su, J., Lyu, J., Ma, Y. & Dou, D. (2024).
SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large
Turbine Array. *Scientific Data*, 11, 649.

[3] Austin, P. C. (2009). Balance diagnostics for comparing the
distribution of baseline covariates between treatment groups in
propensity-score matched samples. *Statistics in Medicine*, 28(25),
3083-3107.

[4] Hüllermeier, E. & Waegeman, W. (2021). Aleatoric and epistemic
uncertainty in machine learning: an introduction to concepts and
methods. *Machine Learning*, 110, 457-506.

[5] Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and
Scalable Predictive Uncertainty Estimation using Deep Ensembles.
*NeurIPS*.

[6] Kendall, A. & Gal, Y. (2017). What Uncertainties Do We Need in
Bayesian Deep Learning for Computer Vision? *NeurIPS*.
