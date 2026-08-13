""" Compute advanced characterization metrics for WindFarm.

Selection bias, aleatoric uncertainty, epistemic uncertainty - the three
advanced metrics from info.docx that apply to a forecasting task like this
one (interpolation threshold / smooth function threshold don't map onto a
simple ridge baseline or dataset-level characterization, so skipped).

Example usage:

	$ python scripts/advanced_metrics_windfarm.py odd_time_predict48h

"""
import os
import sys
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from aigrids import load

import utils
from windfarm_features import records_to_arrays, drop_nan_labels

ARG = sys.argv[1] if len(sys.argv) > 1 else 'odd_time_predict48h'
PATH_CONFIG = 'config.yml'
DATA_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1
RIDGE_ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
SEED = 0
N_BOOTSTRAP = 20
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


def aleatoric_uncertainty(last_known_power, y_test, n_bins=N_BINS):
	""" Irreducible noise: bin test records by their most recent known
	power output, measure variance of actual outcomes *within* each bin,
	average across bins. Given a similar starting condition, how much
	does the future still vary just from real-world randomness?
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


def epistemic_uncertainty(X_train, y_train, X_test, alpha, n_bootstrap=N_BOOTSTRAP, seed=SEED):
	""" Uncertainty from limited training data: bootstrap-resample the
	training set, refit ridge each time, measure how much predictions on
	the same test points vary across refits.
	"""
	rng = np.random.default_rng(seed)
	n_train = len(X_train)
	predictions = []

	for _ in range(n_bootstrap):
		idx = rng.integers(0, n_train, size=n_train)
		model = Pipeline([
			("impute", SimpleImputer(strategy="mean")),
			("scale", StandardScaler()),
			("ridge", Ridge(alpha=alpha)),
		])
		model.fit(X_train[idx], y_train[idx])
		predictions.append(model.predict(X_test).mean(axis=1))

	predictions = np.stack(predictions)  # [n_bootstrap, n_test]
	per_point_variance = predictions.var(axis=0)

	return float(per_point_variance.mean())


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
		"ridge_alpha": RIDGE_ALPHA,
		"n_train": int(X_train.shape[0]),
		"n_test": int(X_test.shape[0]),
		"selection_bias": selection_bias(y_train, y_test),
		"aleatoric_uncertainty": aleatoric_uncertainty(last_known_power, y_test),
		"epistemic_uncertainty": epistemic_uncertainty(
			X_train, y_train, X_test, alpha=RIDGE_ALPHA
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
