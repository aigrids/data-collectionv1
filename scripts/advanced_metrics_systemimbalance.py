""" Compute advanced task and dataset characterization metrics for SystemImbalance.

Computes the model size thresholds and dataset scores of Aryandoust et al.
(2023), arXiv:2311.06876, in the same order of steps as their analysis:
standardize features and labels with training statistics, compute the IO-score
on all data, reduce features with a PCA that keeps 99% of the variance, then
compute STood-, SImb- and Outlier-scores on the reduced features and on labels.

Additionally computes the STood-score for random splits of the same sizes as a
reference for no distribution shift, and the standardized mean difference of
labels between training and testing data.

Example usage:

	$ python scripts/advanced_metrics_systemimbalance.py odd_time_volume_predict1h

"""
import os
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from aigrids import load

import utils
import scores

ARG = sys.argv[1] if len(sys.argv) > 1 else 'odd_time_volume_predict1h'
PATH_CONFIG = 'config.yml'
DATA_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1
N_RANDOM_SPLITS = 3
SEED = 0


def main():
	cfg = utils.parse_config(PATH_CONFIG)

	print(f"Loading SystemImbalance/{ARG} (data_frac={DATA_FRAC}) ...")
	taskdata = load.load_task(
		task_name='SystemImbalance',
		subtask_name=ARG,
		root_path=cfg['root_path_datasets'],
		data_frac=DATA_FRAC
	)

	X_train, Y_train, seq_cols = records_to_arrays(taskdata['train_data'])
	X_val,   Y_val,   _ = records_to_arrays(taskdata['val_data'])
	X_test,  Y_test,  _ = records_to_arrays(taskdata['test_data'])
	print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

	# Thresholds are computed from all data points and all features.
	n = len(X_train) + len(X_val) + len(X_test)
	share_train = len(X_train) / n
	dim_features, dim_labels = X_train.shape[1], Y_train.shape[1]

	results = {
		"subtask_name": ARG,
		"data_frac": DATA_FRAC,
		"n": n,
		"split": [len(X_train) / n, len(X_val) / n, len(X_test) / n],
		"dim_features": dim_features,
		"dim_labels": dim_labels,
		"interpolation_threshold": scores.interpolation_threshold(
			n, share_train, dim_labels),
		"smooth_function_threshold": scores.smooth_function_threshold(
			n, share_train, dim_labels, dim_features)
	}

	# Scores need complete data. Series that are missing in more than 5% of
	# the data points of a split are not used, then incomplete data points.
	feat0 = taskdata['train_data'][0]['features']
	lengths = [len(feat0[c]) for c in seq_cols]
	bounds = np.concatenate([[0], np.cumsum(lengths)])
	used_cols = [
		c for i, c in enumerate(seq_cols) if all(
			np.isnan(X[:, bounds[i]:bounds[i + 1]]).any(axis=1).mean() <= 0.05
			for X in (X_train, X_val, X_test)
		)
	]
	col_mask = np.repeat([c in used_cols for c in seq_cols], lengths)
	X_train, Y_train = drop_incomplete(X_train[:, col_mask], Y_train)
	X_val,   Y_val   = drop_incomplete(X_val[:, col_mask], Y_val)
	X_test,  Y_test  = drop_incomplete(X_test[:, col_mask], Y_test)
	results["series_used_for_scores"] = used_cols
	results["n_used_for_scores"] = [len(X_train), len(X_val), len(X_test)]

	n = len(X_train) + len(X_val) + len(X_test)
	results["scores"] = compute_scores(X_train, Y_train, X_val, Y_val, X_test, Y_test)
	print(json.dumps(results, indent=2))

	# Reference for no distribution shift: random splits of the same sizes.
	rng = np.random.default_rng(SEED)
	X_all = np.concatenate([X_train, X_val, X_test])
	Y_all = np.concatenate([Y_train, Y_val, Y_test])
	i1, i2 = len(X_train), len(X_train) + len(X_val)
	random_splits = []
	for _ in range(N_RANDOM_SPLITS):
		idx = rng.permutation(n)
		tr, va, te = idx[:i1], idx[i1:i2], idx[i2:]
		s = compute_scores(
			X_all[tr], Y_all[tr], X_all[va], Y_all[va], X_all[te], Y_all[te],
			only_stood=True
		)
		random_splits.append(s)
	results["scores_random_split"] = {
		key: float(np.mean([s[key] for s in random_splits])) for key in random_splits[0]
	}

	# Standardized mean difference of the mean label of a data point between
	# training and testing data, with pooled population variance.
	y_tr, y_te = Y_train.mean(axis=1), Y_test.mean(axis=1)
	sigma_pooled = np.sqrt((y_tr.var() + y_te.var()) / 2)
	results["label_standardized_mean_difference"] = float(
		abs(y_tr.mean() - y_te.mean()) / sigma_pooled)

	print(json.dumps(results, indent=2))

	path_results_root = os.path.join(cfg['root_path_results'], 'analysis')
	Path(path_results_root).mkdir(parents=True, exist_ok=True)
	path_out = os.path.join(
		path_results_root, f'advanced_metrics_SystemImbalance_{ARG}.json')

	with open(path_out, 'w') as filesave:
		json.dump(results, filesave, indent=2)

	print(f"Saved results to {path_out}")


def records_to_arrays(records):
	""" One row per data point, with all series next to each other.
	Wind offshore forecasts do not exist for control zones without wind
	offshore generation and are set to zero. Returns (X, Y, series names).
	"""
	feat0 = records[0]["features"]
	seq_cols = [c for c in feat0.keys() if isinstance(feat0[c], list)]
	label_col = list(records[0]["label"].keys())[0]

	X = np.array([
		np.concatenate([
			np.nan_to_num(rec["features"][c], nan=0.) if c == 'wind_off_fc'
			else rec["features"][c] for c in seq_cols
		]) for rec in records
	], dtype=np.float64)
	Y = np.array([rec["label"][label_col] for rec in records], dtype=np.float64)

	return X, Y, seq_cols


def drop_incomplete(X, Y):
	""" Do not use data points with any missing value. """
	keep = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
	if not keep.all():
		print(f"Not using {(~keep).sum()}/{len(keep)} data points with missing values.")

	return X[keep], Y[keep]


def compute_scores(X_train, Y_train, X_val, Y_val, X_test, Y_test, only_stood=False):
	""" Standardize, compute IO-score, reduce features with PCA and compute
	the remaining scores.
	"""
	X_scaler, Y_scaler = StandardScaler(), StandardScaler()
	X_train, Y_train = X_scaler.fit_transform(X_train), Y_scaler.fit_transform(Y_train)
	X_val, Y_val = X_scaler.transform(X_val), Y_scaler.transform(Y_val)
	X_test, Y_test = X_scaler.transform(X_test), Y_scaler.transform(Y_test)

	results = {}
	if not only_stood:
		results["io_score"], _ = scores.io_score(
			np.concatenate([X_train, X_val, X_test]),
			np.concatenate([Y_train, Y_val, Y_test])
		)

	pca = PCA(n_components=0.99).fit(X_train)
	X_train, X_val, X_test = pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)
	results["n_pca_components"] = int(pca.n_components_)

	results["stood_score_val"], _ = scores.stood_score(X_train, X_val)
	results["stood_score_test"], _ = scores.stood_score(X_train, X_test)
	results["stood_score_label_val"], _ = scores.stood_score(Y_train, Y_val)
	results["stood_score_label_test"], _ = scores.stood_score(Y_train, Y_test)
	if only_stood:
		return results

	for name, X, Y in [('train', X_train, Y_train), ('val', X_val, Y_val), ('test', X_test, Y_test)]:
		results[f"simb_score_{name}"], _ = scores.simb_score(X)
		results[f"simb_score_label_{name}"], _ = scores.simb_score(Y)
		results[f"outlier_score_{name}"], _ = scores.outlier_score(X)
	results["outlier_score"], _ = scores.outlier_score(np.concatenate([X_train, X_val, X_test]))

	return results


if __name__ == '__main__':
	main()
