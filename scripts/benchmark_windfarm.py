""" Train and evaluate the WindFarm Ridge regression baseline.

Predicts each turbine's active power output (Patv) over the sub-task's
forecasting horizon (48h or 72h) from its historic SCADA/weather window
and static position (x, y, elevation).

Example usage:

	$ python scripts/benchmark_windfarm.py odd_time_predict48h

"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from aigrids import load

import utils

ARG = sys.argv[1] if len(sys.argv) > 1 else 'odd_time_predict48h'
PATH_CONFIG = 'config.yml'
DATA_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1
RIDGE_ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
SEED = 0

STATIC_COLS = ["x", "y", "Ele"]


def records_to_arrays(records):
	""" Summarize each historic sequence to (last value, mean, trend)
	instead of flattening the full 144-step window - cuts collinearity and
	dimensionality (2307 -> ~51 features) so the linear baseline has a real
	shot at extracting signal instead of drowning in redundant, near-
	identical consecutive readings. Returns (X, y) numpy arrays.
	"""
	if not records:
		return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

	feat0 = records[0]["features"]
	seq_cols = [c for c in feat0.keys() if c not in ("Tmstamp", *STATIC_COLS)]

	n_records = len(records)
	n_features = len(STATIC_COLS) + len(seq_cols) * 3  # last, mean, trend
	label_len = len(records[0]["label"]["Patv"])

	X = np.empty((n_records, n_features), dtype=np.float32)
	y = np.empty((n_records, label_len), dtype=np.float32)

	for i, rec in enumerate(records):
		feat = rec["features"]
		static_vals = [feat[c] for c in STATIC_COLS]
		seq_summary = []
		for c in seq_cols:
			seq = np.asarray(feat[c], dtype=np.float32)
			seq_summary.extend([seq[-1], np.nanmean(seq), seq[-1] - seq[0]])
		X[i] = np.concatenate([static_vals, seq_summary])
		y[i] = rec["label"]["Patv"]

	return X, y


def drop_nan_labels(X, y):
	""" Drop records whose label window contains any NaN (unusable as
	ground truth, and would corrupt evaluation if imputed instead).
	"""
	keep_mask = ~np.isnan(y).any(axis=1)
	n_dropped = len(y) - keep_mask.sum()
	if n_dropped:
		print(f"Dropping {n_dropped}/{len(y)} records with NaN in label window.")
	return X[keep_mask], y[keep_mask]


def main():
	cfg = utils.parse_config(PATH_CONFIG)

	print(f"Loading WindFarm/{ARG} (data_frac={DATA_FRAC}) ...")
	taskdata = load.load_task(
		task_name='WindFarm',
		subtask_name=ARG,
		root_path=cfg['root_path_datasets'],
		data_frac=DATA_FRAC
	)

	X_train, y_train = records_to_arrays(taskdata['train_data'])
	X_val,   y_val   = records_to_arrays(taskdata['val_data'])
	X_test,  y_test  = records_to_arrays(taskdata['test_data'])

	X_train, y_train = drop_nan_labels(X_train, y_train)
	X_val,   y_val   = drop_nan_labels(X_val, y_val)
	X_test,  y_test  = drop_nan_labels(X_test, y_test)

	print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

	model = Pipeline([
		("impute", SimpleImputer(strategy="mean")),
		("scale", StandardScaler()),
		("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED)),
	])

	t0 = time.perf_counter()
	model.fit(X_train, y_train)
	train_time_sec = time.perf_counter() - t0

	t0 = time.perf_counter()
	pred_val = model.predict(X_val)
	pred_test = model.predict(X_test)
	infer_time_sec = time.perf_counter() - t0

	# Trivial baseline: always predict the training mean, ignoring inputs
	# entirely. Ridge should clearly beat this - if it doesn't (or barely
	# does), that's a sign alpha is so high the model has stopped using
	# the features at all, not that it's "doing well".
	mean_pred_val = np.tile(y_train.mean(axis=0), (len(y_val), 1))
	mean_pred_test = np.tile(y_train.mean(axis=0), (len(y_test), 1))

	results = {
		"subtask_name": ARG,
		"data_frac": DATA_FRAC,
		"ridge_alpha": RIDGE_ALPHA,
		"n_train": int(X_train.shape[0]),
		"n_val": int(X_val.shape[0]),
		"n_test": int(X_test.shape[0]),
		"n_features": int(X_train.shape[1]),
		"n_label_steps": int(y_train.shape[1]),
		"train_time_sec": train_time_sec,
		"infer_time_sec": infer_time_sec,
		"val_mae": float(mean_absolute_error(y_val, pred_val)),
		"val_rmse": float(mean_squared_error(y_val, pred_val) ** 0.5),
		"test_mae": float(mean_absolute_error(y_test, pred_test)),
		"test_rmse": float(mean_squared_error(y_test, pred_test) ** 0.5),
		"mean_baseline_val_mae": float(mean_absolute_error(y_val, mean_pred_val)),
		"mean_baseline_val_rmse": float(mean_squared_error(y_val, mean_pred_val) ** 0.5),
		"mean_baseline_test_mae": float(mean_absolute_error(y_test, mean_pred_test)),
		"mean_baseline_test_rmse": float(mean_squared_error(y_test, mean_pred_test) ** 0.5),
	}

	print(json.dumps(results, indent=2))

	path_results_root = os.path.join(cfg['root_path_results'], 'benchmark')
	Path(path_results_root).mkdir(parents=True, exist_ok=True)
	alpha_tag = str(RIDGE_ALPHA).replace('.', 'p')
	path_out = os.path.join(path_results_root, f'ridge_windfarm_{ARG}_alpha{alpha_tag}.json')

	with open(path_out, 'w') as filesave:
		json.dump(results, filesave, indent=2)

	print(f"Saved results to {path_out}")


if __name__ == '__main__':
	main()
