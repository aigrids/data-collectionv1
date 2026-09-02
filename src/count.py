""" Counts data points and shares

"""


def datapoints(ds: dict):
	""" """

	n_train = len(ds['train_data'])
	n_val = len(ds['val_data'])
	n_test = len(ds['test_data'])

	n_datapoints = n_train + n_val + n_test

	train_val_test_split = (
		n_train / n_datapoints, 
		n_val / n_datapoints, 
		n_test / n_datapoints
	)

	# entities = distinct turbines, identified by their (x, y) position
	# since raw turbine IDs aren't included in the standardized record
	all_records = ds['train_data'] + ds['val_data'] + ds['test_data']
	positions = {(r['features']['x'], r['features']['y']) for r in all_records}
	n_entities = len(positions)

	# dim_features/dim_labels: raw dimensionality as delivered by the task
	# (static fields + one value per historic timestep per sequence column),
	# not any model-specific feature engineering done downstream of this.
	feat0 = all_records[0]['features']
	static_cols = ['x', 'y', 'Ele']
	seq_cols = [c for c in feat0.keys() if c not in ('Tmstamp', *static_cols)]
	hist_len = len(feat0[seq_cols[0]])
	dim_features = len(static_cols) + len(seq_cols) * hist_len
	dim_labels = len(all_records[0]['label']['Patv'])

	results_count = {
		"n_datapoints": n_datapoints,
		"train_val_test_split": train_val_test_split,
		"n_entities": n_entities,
		"dim_features": dim_features,
		"dim_labels": dim_labels
	}

	return results_count