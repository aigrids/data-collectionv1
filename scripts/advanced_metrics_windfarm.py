""" Dataset-level advanced characterization metrics for WindFarm.

- Train-test distribution shift + a simple binned aleatoric-uncertainty
  baseline
- Aleatoric/epistemic/total uncertainty itself is computed in
  probabilistic_dmst_windfarm.py, not here

selection_bias
    Standardized mean difference, train vs. test target means.

aleatoric_uncertainty_binned_proxy
    Within-bin variance of outcomes, binned by most recent known power.

Example usage:

	$ python scripts/advanced_metrics_windfarm.py odd_time_predict48h

"""
import os
import sys
import json

import numpy as np
from sklearn.impute import SimpleImputer

from aigrids import load

import utils
from windfarm_features import records_to_arrays, drop_nan_labels

ARG = sys.argv[1] if len(sys.argv) > 1 else 'odd_time_predict48h'
PATH_CONFIG = 'config.yml'
DATA_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1
N_BINS = 10


def selection_bias(y_train, y_test):
	""" Standardized mean difference between train and test target
	distributions: |mean_train - mean_test| / pooled_std. Measures how
	much the split shifts what's being predicted - relevant here since
	odd_time/odd_space splits are deliberately built around a shift.
	"""
	train_vals = y_train.mean(axis=1)  # per-record mean forecast value
	test_vals = y_test.mean(axis=1)

	mean_diff = abs(train_vals.mean() - test_vals.mean())
	pooled_std = np.sqrt((train_vals.var() + test_vals.var()) / 2)

	return float(mean_diff / pooled_std) if pooled_std > 0 else float("nan")


def aleatoric_uncertainty_binned_proxy(last_known_power, y_test, n_bins=N_BINS):
	""" Empirical proxy (not from H&W, see module docstring): bin test
	records by their most recent known power output, measure variance of
	actual outcomes *within* each bin, average across bins.
	"""
	target = y_test.mean(axis=1)

	bin_edges = np.quantile(last_known_power, np.linspace(0, 1, n_bins + 1))
	bin_edges[-1] += 1e-6  # include max value in the last bin
	bin_idx = np.digitize(last_known_power, bin_edges[1:-1])

	bin_variances, bin_weights = [], []
	for b in range(n_bins):
		mask = bin_idx == b
		if mask.sum() > 1:
			bin_variances.append(target[mask].var())
			bin_weights.append(mask.sum())

	if not bin_variances:
		return float("nan")

	return float(np.average(bin_variances, weights=bin_weights))


def main():
	cfg = utils.parse_config(PATH_CONFIG)

	print(f"Loading WindFarm/{ARG} (data_frac={DATA_FRAC}) ...")
	taskdata = load.load_task(
		task_name="WindFarm",
		subtask_name=ARG,
		root_path=cfg["root_path_datasets"],
		data_frac=DATA_FRAC
	)

	X_train, y_train, feature_names = records_to_arrays(taskdata["train_data"])
	X_test,  y_test,  _             = records_to_arrays(taskdata["test_data"])

	X_train, y_train = drop_nan_labels(X_train, y_train)
	X_test,  y_test  = drop_nan_labels(X_test, y_test)

	print(f"train: {X_train.shape}, test: {X_test.shape}")

	last_power_idx = feature_names.index("Patv_last")
	imputer = SimpleImputer(strategy="mean")
	X_test_imputed = imputer.fit(X_train).transform(X_test)
	last_known_power = X_test_imputed[:, last_power_idx]

	results = {
		"subtask_name": ARG,
		"data_frac": DATA_FRAC,
		"n_train": int(X_train.shape[0]),
		"n_test": int(X_test.shape[0]),
		"selection_bias": selection_bias(y_train, y_test),
		"aleatoric_uncertainty_binned_proxy": aleatoric_uncertainty_binned_proxy(
			last_known_power, y_test
		),
	}

	print(json.dumps(results, indent=2))

	path_results_root = os.path.join(cfg["root_path_results"], "advanced_metrics")
	os.makedirs(path_results_root, exist_ok=True)
	path_out = os.path.join(path_results_root, f"advanced_metrics_windfarm_{ARG}.json")

	with open(path_out, "w") as filesave:
		json.dump(results, filesave, indent=2)

	print(f"Saved results to {path_out}")


if __name__ == "__main__":
	main()
