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

	results_count = {
		"n_datapoints": n_datapoints,
		"train_val_test_split": train_val_test_split
	}

	return results_count