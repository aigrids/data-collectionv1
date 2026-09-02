""" Compute descriptive statistics for the raw WindFarm dataset.

Example usage:

	$ python scripts/dataset_stats_windfarm.py

"""
import os
import json

import pandas as pd
import numpy as np

import utils

PATH_CONFIG = 'config.yml'


def main():
	cfg = utils.parse_config(PATH_CONFIG)
	root = os.path.join(cfg['root_path_datasets'], 'WindFarm')

	path_scada = os.path.join(root, 'sdwpf_2001_2112_full.csv')
	path_loc = os.path.join(root, 'sdwpf_turb_location_elevation.csv')

	scada = pd.read_csv(path_scada)
	loc = pd.read_csv(path_loc)

	scada['Tmstamp'] = pd.to_datetime(scada['Tmstamp'])

	numeric_cols = [
		c for c in scada.columns
		if c not in ('TurbID', 'Tmstamp') and pd.api.types.is_numeric_dtype(scada[c])
	]

	stats = {
		'n_rows': int(len(scada)),
		'n_turbines': int(scada['TurbID'].nunique()),
		'date_min': str(scada['Tmstamp'].min()),
		'date_max': str(scada['Tmstamp'].max()),
		'n_days': int((scada['Tmstamp'].max() - scada['Tmstamp'].min()).days),
		'expected_rows_if_complete': int(
			scada['TurbID'].nunique()
			* ((scada['Tmstamp'].max() - scada['Tmstamp'].min()).total_seconds() / 600 + 1)
		),
		'missing_rate_pct_by_col': {
			c: round(float(scada[c].isna().mean() * 100), 3) for c in numeric_cols
		},
		'column_stats': {
			c: {
				'mean': round(float(scada[c].mean()), 4),
				'std': round(float(scada[c].std()), 4),
				'min': round(float(scada[c].min()), 4),
				'max': round(float(scada[c].max()), 4),
				'p50': round(float(scada[c].median()), 4),
			}
			for c in numeric_cols
		},
		'negative_patv_rate_pct': round(float((scada['Patv'] < 0).mean() * 100), 3),
		'zero_patv_rate_pct': round(float((scada['Patv'] == 0).mean() * 100), 3),
		'location': {
			'n_turbines': int(len(loc)),
			'x_range': [round(float(loc['x'].min()), 2), round(float(loc['x'].max()), 2)],
			'y_range': [round(float(loc['y'].min()), 2), round(float(loc['y'].max()), 2)],
			'elevation_range': [round(float(loc['Ele'].min()), 2), round(float(loc['Ele'].max()), 2)],
		},
	}

	print(json.dumps(stats, indent=2))

	path_out = os.path.join(cfg['root_path_results'], 'windfarm_dataset_stats.json')
	os.makedirs(os.path.dirname(path_out), exist_ok=True)
	with open(path_out, 'w') as f:
		json.dump(stats, f, indent=2)

	print(f"\nSaved to {path_out}")


if __name__ == '__main__':
	main()
