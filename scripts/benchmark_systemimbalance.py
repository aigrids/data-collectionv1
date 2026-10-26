""" Train and evaluate the SystemImbalance gradient boosted tree baseline.

Predicts the system imbalance of a control zone, or the balancing energy
price in Germany, for every quarter hour of the next hour from the published
system imbalance of the zone and of the other control zones (for the price:
of Germany, and the balancing energy price) of the past 24 hours, and from
day-ahead forecasts and prices. Model, day-ahead features and
hyper-parameter grid follow Kruse et al., "Secondary
control activation analysed and predicted with explainable AI" (2022),
https://github.com/johkruse/XAI-for-aFRR-activation (MIT license).
Different from the original, the historic system imbalance is used as input, and
hyper-parameters are selected on the standardized validation split instead
of a random cross-validation.

Example usage:

	$ python scripts/benchmark_systemimbalance.py odd_time_volume_predict1h

"""
import os
import sys
import json
import time
import itertools
from pathlib import Path

import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor

from aigrids import load

import utils
from systemimbalance_features import records_to_arrays

ARG = sys.argv[1] if len(sys.argv) > 1 else 'odd_time_volume_predict1h'
PATH_CONFIG = 'config.yml'
DATA_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1
# Number of randomly drawn hyper-parameter combinations. 0 uses the full grid.
N_SEARCH = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SEED = 0
N_JOBS = 8
QUANTILE = 0.9

# Hyper-parameter grid of Kruse et al.
PARAMS_GRID = {
	'num_leaves': [10, 30, 200, 900],
	'max_depth': [10, 5],
	'subsample': [1, 0.9, 0.5],
	'learning_rate': [0.001, 0.05, 0.01, 0.1],
	'min_child_samples': [50, 100, 500]
}

# Features that are not perturbed for measuring perturbation robustness.
CALENDAR_FEATURES = ['hour', 'weekday', 'step']


def main():
	cfg = utils.parse_config(PATH_CONFIG)

	print(f"Loading SystemImbalance/{ARG} (data_frac={DATA_FRAC}) ...")
	taskdata = load.load_task(
		task_name='SystemImbalance',
		subtask_name=ARG,
		root_path=cfg['root_path_datasets'],
		data_frac=DATA_FRAC
	)

	X_train, y_train, feature_names, n_steps = records_to_arrays(taskdata['train_data'])
	X_val,   y_val,   _, _ = records_to_arrays(taskdata['val_data'])
	X_test,  y_test,  _, _ = records_to_arrays(taskdata['test_data'])
	print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

	# missing labels are not used for training and evaluation
	m_train, m_val, m_test = ~np.isnan(y_train), ~np.isnan(y_val), ~np.isnan(y_test)

	results = {
		"subtask_name": ARG,
		"data_frac": DATA_FRAC,
		"n_search": N_SEARCH,
		"n_train": int(len(taskdata['train_data'])),
		"n_val": int(len(taskdata['val_data'])),
		"n_test": int(len(taskdata['test_data'])),
		"n_features": len(feature_names),
		"n_steps": n_steps
	}

	# Trivial baselines: always predict the training mean, or the mean daily
	# profile of training data (both from Kruse et al.).
	i_hour = feature_names.index('hour')
	hour_train, hour_test = X_train[m_train, i_hour], X_test[m_test, i_hour]
	profile = {h: y_train[m_train][hour_train == h].mean() for h in np.unique(hour_train)}
	train_mean = y_train[m_train].mean()
	pred_mean = np.full(m_test.sum(), train_mean)
	pred_profile = np.array([profile.get(h, train_mean) for h in hour_test])
	results["test_metrics_mean_predictor"] = point_metrics(y_test[m_test], pred_mean)
	results["test_metrics_daily_profile"] = point_metrics(y_test[m_test], pred_profile)

	# Persistence: repeat the last published value of the label.
	label_col = list(taskdata['train_data'][0]['label'].keys())[0]
	if f'{label_col}_lag1' in feature_names:
		pred_persistence = X_test[:, feature_names.index(f'{label_col}_lag1')]
		m_pers = m_test & ~np.isnan(pred_persistence)
		results["test_metrics_persistence"] = point_metrics(
			y_test[m_pers], pred_persistence[m_pers])

	for objective in ['l2', 'quantile']:
		print(f"\nHyper-parameter search for objective {objective} ...")
		best_params = search_params(
			X_train[m_train], y_train[m_train], X_val[m_val], y_val[m_val], objective
		)
		print("Best parameters:", best_params)

		model = make_model(objective, best_params)
		t0 = time.perf_counter()
		model.fit(X_train[m_train], y_train[m_train])
		train_time_sec = time.perf_counter() - t0

		pred_val, pred_test = model.predict(X_val), model.predict(X_test)
		if objective == 'l2':
			results["l2"] = {
				"best_params": best_params,
				"train_time_sec": train_time_sec,
				"val_metrics": point_metrics(y_val[m_val], pred_val[m_val]),
				"test_metrics": point_metrics(y_test[m_test], pred_test[m_test]),
				"test_metrics_per_step": [
					point_metrics(
						y_test[step::n_steps][m_test[step::n_steps]],
						pred_test[step::n_steps][m_test[step::n_steps]])
					for step in range(n_steps)
				],
				"computation_time": computation_time(
					model, X_test, n_steps, train_time_sec),
				"batch_scaling_factor": batch_scaling_factor(model, X_test, n_steps),
				"perturbation_robustness": perturbation_robustness(
					model, X_train[m_train], X_test, n_steps, feature_names),
				"training_data_efficiency": training_data_efficiency(
					best_params, X_train, y_train, X_test, y_test, n_steps)
			}
		else:
			results["quantile"] = {
				"alpha": QUANTILE,
				"best_params": best_params,
				"train_time_sec": train_time_sec,
				"val_metrics": quantile_metrics(y_val[m_val], pred_val[m_val]),
				"test_metrics": quantile_metrics(y_test[m_test], pred_test[m_test]),
				"test_metrics_train_quantile": quantile_metrics(
					y_test[m_test],
					np.full(m_test.sum(), np.quantile(y_train[m_train], QUANTILE)))
			}

	print(json.dumps(results, indent=2))

	path_results_root = os.path.join(cfg['root_path_results'], 'benchmark')
	Path(path_results_root).mkdir(parents=True, exist_ok=True)
	path_out = os.path.join(
		path_results_root, f'benchmark_results_SystemImbalance_{ARG}.json')

	with open(path_out, 'w') as filesave:
		json.dump(results, filesave, indent=2)

	print(f"Saved results to {path_out}")


def make_model(objective, params, n_estimators=None):
	""" Gradient boosted tree model with the fixed settings of Kruse et al. """
	params = dict(params)
	if n_estimators is None:
		n_estimators = params.pop('n_estimators', 1000)
	return LGBMRegressor(
		n_estimators=n_estimators, objective=objective, alpha=QUANTILE,
		min_child_weight=0, subsample_freq=1, n_jobs=N_JOBS,
		random_state=SEED, verbose=-1, **params
	)


def search_params(X_train, y_train, X_val, y_val, objective):
	""" Select hyper-parameters and number of trees by early stopping on
	validation data.
	"""
	combos = [
		dict(zip(PARAMS_GRID.keys(), values))
		for values in itertools.product(*PARAMS_GRID.values())
	]
	if 0 < N_SEARCH < len(combos):
		rng = np.random.default_rng(SEED)
		combos = [combos[i] for i in rng.choice(len(combos), N_SEARCH, replace=False)]

	best_score, best_params = np.inf, None
	for i, params in enumerate(combos):
		model = make_model(objective, params, n_estimators=1000)
		model.fit(
			X_train, y_train, eval_set=[(X_val, y_val)],
			callbacks=[lgb.early_stopping(20, verbose=False)]
		)
		score = list(model.best_score_['valid_0'].values())[0]
		if score < best_score:
			best_score = score
			best_params = {**params, 'n_estimators': int(model.best_iteration_)}
		print(f"  {i + 1}/{len(combos)} val score {score:.2f} (best {best_score:.2f})")

	return best_params


def point_metrics(y_true, y_pred):
	""" MAE, RMSE, R^2 and share of correctly predicted signs. """
	errors = y_pred - y_true
	ss_tot = np.sum((y_true - y_true.mean()) ** 2)
	return {
		"mae": float(np.mean(np.abs(errors))),
		"rmse": float(np.sqrt(np.mean(errors ** 2))),
		"r2": float(1 - np.sum(errors ** 2) / ss_tot),
		"sign_accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true)))
	}


def quantile_metrics(y_true, y_pred):
	""" Mean quantile loss and share of labels below the predicted quantile. """
	residual = y_true - y_pred
	return {
		"quantile_loss": float(np.mean(residual * (QUANTILE - (residual < 0)))),
		"coverage": float(np.mean(y_true <= y_pred))
	}


def computation_time(model, X_test, n_steps, train_time_sec, n_infer_samples=500):
	""" AI.grids v1 Table 2: training time and average inference time per
	data point.
	"""
	n_points = min(n_infer_samples, len(X_test) // n_steps)
	infer_times = []
	for i in range(n_points):
		x = X_test[i * n_steps:(i + 1) * n_steps]
		t0 = time.perf_counter()
		model.predict(x)
		infer_times.append(time.perf_counter() - t0)

	return {
		"train_time_s": float(train_time_sec),
		"avg_infer_time_per_data_point_ms": float(np.mean(infer_times) * 1000)
	}


def batch_scaling_factor(model, X_test, n_steps, batch_sizes=(4, 8, 16, 32, 64),
		n_points=256, n_repeats=3):
	""" AI.grids v1 Table 2: batch scaling factor = T_b / (b * T_1). """
	n_points = min(n_points, len(X_test) // n_steps)

	def time_batch(batch_size):
		times = []
		for _ in range(n_repeats):
			t0 = time.perf_counter()
			for i in range(0, n_points, batch_size):
				model.predict(X_test[i * n_steps:(i + batch_size) * n_steps])
			times.append(time.perf_counter() - t0)
		return float(np.mean(times))

	t1_total = time_batch(1)

	results = {}
	for b in batch_sizes:
		tb = time_batch(b)
		results[f"batch_{b}"] = {"total_time_s": tb, "scaling_factor": tb / t1_total}
	return results


def perturbation_robustness(model, X_train, X_test, n_steps, feature_names,
		sigmas=(0.01, 0.05, 0.1), n_eval=300):
	""" AI.grids v1 Table 2: perturbation robustness
	= (1/(N*sigma)) * sum ||y - y'||_2.

	Noise is scaled with the standard deviation of each feature in training
	data. Calendar features are not perturbed.
	"""
	rng = np.random.default_rng(SEED)
	n_points = min(n_eval, len(X_test) // n_steps)
	X = X_test[:n_points * n_steps]
	std = np.nanstd(X_train, axis=0)
	std[[feature_names.index(c) for c in CALENDAR_FEATURES]] = 0

	out_clean = model.predict(X).reshape(n_points, n_steps)
	results = {}
	for sigma in sigmas:
		noise = rng.standard_normal(X.shape).astype(np.float32) * std * sigma
		out_noisy = model.predict(X + noise).reshape(n_points, n_steps)
		diffs = np.linalg.norm(out_clean - out_noisy, axis=1)
		results[f"sigma_{sigma}"] = float(np.mean(diffs) / sigma)
	return results


def training_data_efficiency(params, X_train, y_train, X_test, y_test, n_steps,
		fractions=(0.05, 0.1, 0.25, 0.5, 1.0)):
	""" AI.grids v1 Table 2: training data efficiency curve. """
	rng = np.random.default_rng(SEED)
	n_points = len(X_train) // n_steps
	m_test = ~np.isnan(y_test)

	results = {}
	for frac in fractions:
		n = int(n_points * frac)
		points = rng.choice(n_points, size=n, replace=False)
		rows = (points[:, None] * n_steps + np.arange(n_steps)).reshape(-1)
		rows = rows[~np.isnan(y_train[rows])]

		model = make_model('l2', params).fit(X_train[rows], y_train[rows])
		metrics = point_metrics(y_test[m_test], model.predict(X_test[m_test]))
		results[f"frac_{frac}"] = {"n_samples": n, "rmse": metrics["rmse"]}
	return results


if __name__ == '__main__':
	main()
