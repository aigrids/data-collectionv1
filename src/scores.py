""" Dataset scores and model size thresholds for characterizing tasks.

The four dataset scores are a numpy port of the functions in
https://github.com/ArsamAryandoust/EnergyTransitionTasks/blob/master/src/analyse_functions.py
(MIT license), introduced in Aryandoust et al., "Unified machine learning tasks
and datasets for enhancing renewable energy" (2023), arXiv:2311.06876.

All functions expect arrays with one row per data point and one column per
feature or label, without missing values.

"""
import numpy as np

LOG_EPSILON = 1e-39


def interpolation_threshold(n, share_train, dim_labels):
	""" Number of model parameters for reaching the over-parameterized regime. """
	return int(share_train * n * dim_labels)


def smooth_function_threshold(n, share_train, dim_labels, dim_features):
	""" Number of model parameters for learning smooth functions. """
	return int(share_train * n * dim_labels * dim_features)


def remove_outliers_with_tukey_fences(x):
	q1, q3 = np.quantile(x, [0.25, 0.75])
	iqr = q3 - q1
	return x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]


def create_distribution(x, min, max, n_bins=10000):
	histogram, _ = np.histogram(x, bins=n_bins, range=(min, max))
	return histogram / histogram.sum()


def compute_js_divergence(p, q):
	part1 = np.sum(p * np.log2(2.0 * p + LOG_EPSILON) - p * np.log2(p + q + LOG_EPSILON))
	part2 = np.sum(q * np.log2(2.0 * q + LOG_EPSILON) - q * np.log2(q + p + LOG_EPSILON))
	return 0.5 * part1 + 0.5 * part2


def simb_score(dataset, n_bins=10000):
	""" Sample imbalance score: mean Jensen-Shannon divergence between the
	distribution of each column and a uniform distribution.
	"""
	uniform_distribution = np.ones(n_bins) / n_bins
	js_divergences = []
	for feature in range(dataset.shape[1]):
		dataset_slice = remove_outliers_with_tukey_fences(dataset[:, feature])
		distribution = create_distribution(
			dataset_slice, dataset_slice.min(), dataset_slice.max(), n_bins)
		js_divergences.append(compute_js_divergence(distribution, uniform_distribution))

	return float(np.mean(js_divergences)), np.array(js_divergences)


def stood_score(train_set, other_set, n_bins=10000):
	""" Spatio-temporal out-of-distribution score: mean Jensen-Shannon
	divergence between the distributions of each column in two splits.
	"""
	js_divergences = []
	for feature in range(train_set.shape[1]):
		train_slice = remove_outliers_with_tukey_fences(train_set[:, feature])
		other_slice = remove_outliers_with_tukey_fences(other_set[:, feature])
		min = np.minimum(train_slice.min(), other_slice.min())
		max = np.maximum(train_slice.max(), other_slice.max())
		js_divergences.append(compute_js_divergence(
			create_distribution(train_slice, min, max, n_bins),
			create_distribution(other_slice, min, max, n_bins)
		))

	return float(np.mean(js_divergences)), np.array(js_divergences)


def outlier_score(dataset):
	""" Outlier score: mean severity of values beyond the inner Tukey fences,
	scaled linearly from 0 at the inner to 1 at the outer fences.
	"""
	outlier_subscores = []
	for feature in range(dataset.shape[1]):
		x = dataset[:, feature]
		q1, q3 = np.quantile(x, [0.25, 0.75])
		iqr = q3 - q1
		f1, f2 = q1 - 1.5 * iqr, q3 + 1.5 * iqr
		e1, e2 = q1 - 3 * iqr, q3 + 3 * iqr
		n_extreme = np.sum((x <= e1) | (x >= e2))
		left = x[(x <= f1) & (x >= e1)]
		right = x[(x <= e2) & (x >= f2)]
		n_outliers = n_extreme + len(left) + len(right)
		if n_outliers == 0 or iqr == 0:
			outlier_subscores.append(0.)
			continue
		scores_sum = n_extreme + np.sum((left - e1) / (f1 - e1)) + np.sum((right - f2) / (e2 - f2))
		outlier_subscores.append(scores_sum / n_outliers)

	return float(np.mean(outlier_subscores)), np.array(outlier_subscores)


def io_score(features, labels):
	""" Input-output score: mean absolute incremental ratio of each label
	with respect to each feature, mapped to [0, 1] with an arctan.
	"""
	mean_deltas = np.full((features.shape[1], labels.shape[1]), np.nan)
	for feature in range(features.shape[1]):
		order = np.argsort(features[:, feature], kind='stable')
		delta_feature = np.diff(features[order, feature])
		delta_labels = np.diff(labels[order], axis=0)
		with np.errstate(divide='ignore', invalid='ignore'):
			delta = delta_labels / delta_feature[:, None]
		delta[~np.isfinite(delta)] = np.nan
		delta = np.arctan(np.abs(delta)) / (np.pi / 2)
		if not np.all(np.isnan(delta)):
			mean_deltas[feature] = np.nanmean(delta, axis=0)

	return float(np.nanmean(mean_deltas)), mean_deltas
