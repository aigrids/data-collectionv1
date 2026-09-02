""" Shared feature-building helpers for WindFarm scripts.

"""
import warnings

import numpy as np

# some historic windows are fully missing for a given sensor; nanmean on an
# all-NaN slice correctly returns NaN (handled downstream by imputation),
# the warning is just noise
warnings.filterwarnings("ignore", message="Mean of empty slice")

STATIC_COLS = ["x", "y", "Ele"]


def records_to_arrays(records):
	""" Summarize each historic sequence to (last value, mean, trend)
	instead of flattening the full 144-step window - cuts collinearity and
	dimensionality (2307 -> ~51 features) so a linear model has a real
	shot at extracting signal instead of drowning in redundant, near-
	identical consecutive readings. Returns (X, y, feature_names).
	"""
	if not records:
		return (
			np.empty((0, 0), dtype=np.float32),
			np.empty((0, 0), dtype=np.float32),
			[]
		)

	feat0 = records[0]["features"]
	seq_cols = [c for c in feat0.keys() if c not in ("Tmstamp", *STATIC_COLS)]

	feature_names = list(STATIC_COLS)
	for c in seq_cols:
		feature_names.extend([f"{c}_last", f"{c}_mean", f"{c}_trend"])

	n_records = len(records)
	n_features = len(feature_names)
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

	return X, y, feature_names


def drop_nan_labels(X, y):
	""" Drop records whose label window contains any NaN (unusable as
	ground truth, and would corrupt evaluation if imputed instead).
	"""
	keep_mask = ~np.isnan(y).any(axis=1)
	n_dropped = len(y) - keep_mask.sum()
	if n_dropped:
		print(f"Dropping {n_dropped}/{len(y)} records with NaN in label window.")
	return X[keep_mask], y[keep_mask]
