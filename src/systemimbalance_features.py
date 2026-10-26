""" Shared feature-building helpers for SystemImbalance scripts.

Features extend the day-ahead model of Kruse et al., "Secondary control
activation analysed and predicted with explainable AI" (2022), published at
https://github.com/johkruse/XAI-for-aFRR-activation under an MIT license:
day-ahead forecasts and prices of a quarter hour, their ramps with respect to
the previous quarter hour, the hour of the day and the day of the week. In
addition, the historic series of 24 hours, as far as they are published
before the predicted hour, and the position of the quarter hour in the
predicted hour are used.

"""
from datetime import datetime

import numpy as np


def records_to_arrays(records):
	""" Turn each record into one row per predicted quarter hour. Rows of a
	record stay next to each other, so that row i belongs to record
	i // n_steps. Returns (X, y, feature_names, n_steps).
	"""
	feat0 = records[0]["features"]
	label_col = list(records[0]["label"].keys())[0]
	n_steps = len(records[0]["label"][label_col])

	# series of the historic window are longer than the predicted window
	list_cols = [c for c in feat0.keys() if isinstance(feat0[c], list)]
	hist_cols = [c for c in list_cols if len(feat0[c]) != n_steps]
	seq_cols = [c for c in list_cols if len(feat0[c]) == n_steps]

	# history is ordered from oldest to most recent value
	feature_names = []
	for c in hist_cols:
		n_hist = len(feat0[c])
		feature_names.extend([f"{c}_lag{n_hist - i}" for i in range(n_hist)])
	n_hist_total = len(feature_names)
	for c in seq_cols:
		feature_names.extend([c, f"{c}_ramp"])
	feature_names.extend(["hour", "weekday", "step"])

	n_records = len(records)
	X = np.empty((n_records, n_steps, len(feature_names)), dtype=np.float32)
	y = np.empty((n_records, n_steps), dtype=np.float32)
	steps = np.arange(n_steps)

	for i, rec in enumerate(records):
		feat = rec["features"]
		X[i, :, :n_hist_total] = np.concatenate([feat[c] for c in hist_cols])
		for j, c in enumerate(seq_cols):
			seq = np.asarray(feat[c], dtype=np.float32)
			X[i, :, n_hist_total + 2 * j] = seq
			# the first predicted quarter hour has no previous value
			X[i, :, n_hist_total + 2 * j + 1] = np.diff(seq, prepend=np.nan)
		stamp = datetime.fromisoformat(feat["Tmstamp"])
		X[i, :, -3] = (stamp.hour + stamp.minute / 60. + steps / 4.) % 24
		X[i, :, -2] = stamp.weekday()
		X[i, :, -1] = steps
		y[i] = rec["label"][label_col]

	return X.reshape(-1, len(feature_names)), y.reshape(-1), feature_names, n_steps
