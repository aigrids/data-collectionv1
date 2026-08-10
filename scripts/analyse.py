""" Analyse dataset.

Choose arguments from:
	- BE (BuildingElectricity)
	- WF (WindFarm)
	- SC (SolarCube)
	- PG (PowerGraph)
	- OD (OPFData)


Example usage:

	$ python scripts/analyse.py BE 

"""
import os
import sys
import json
from pathlib import Path

from aigrids import load

import utils
import count

ARG = sys.argv[1]
DATA_FRAC = 1
PATH_CONFIG = 'config_arsam.yml'
MAP_ARG_TO_TASK = {
	'BE': 'BuildingElectricity',
	'WF': 'WindFarm',
	'SC': 'SolarCube',
	'PG': 'PowerGraph',
	'OD': 'OPFData'
}

MAP_TASK_TO_AVAILSUBTASKS = {
	'BuildingElectricity': [
		'odd_time_buildings92',
		'odd_space_buildings92',
		'odd_spacetime_buildings92',
		'odd_time_wload_buildings92',
		'odd_space_wload_buildings92',
		'odd_spacetime_wload_buildings92',
		'odd_time_buildings451',
		'odd_space_buildings451',
		'odd_spacetime_buildings451',
		'odd_time_wload_buildings451',
		'odd_spac_wloade_buildings451',
		'odd_spacetime_wload_buildings451'
	],
	'WindFarm': [
		'odd_time_predict48h',
		'odd_space_predict48h',
		'odd_spacetime_predict48h',
		'odd_time_predict72h',
		'odd_space_predict72h',
		'odd_spacetime_predict72h'
	],
	'SolarCube': [
		'odd_time_area',
		'odd_time_point',
		'odd_space_area',
		'odd_space_point',
		'odd_spacetime_area',
		'odd_spacetime_point'
	],
	'PowerGraph': [
		'cascading_failure_binary',
		'cascading_failure_multiclass',
		'demand_not_served',
		'cascading_failure_sequence'
	],
	'OPFData': [
		'train_small_test_medium',
		'train_small_test_large',
		'train_medium_test_small',
		'train_medium_test_large',
		'train_large_test_small',
		'train_large_test_medium'
	]
}


def main():
	cfg = utils.parse_config(PATH_CONFIG)
	task_name = MAP_ARG_TO_TASK[ARG]
	subtask_list = MAP_TASK_TO_AVAILSUBTASKS[task_name]

	# all results to fill
	results_dict = {}

	for subtask_name in subtask_list:

		# subtask results to fill
		results_dict[subtask_name] = {}

		# load dataset
		ds = load.load_task(
			task_name=task_name,
			subtask_name=subtask_name,
			root_path=cfg['root_path_datasets'],
			data_frac=DATA_FRAC
		)

		# get counting results
		results_count = count.datapoints(ds)

		# update subtask results
		results_dict[subtask_name].update(results_count)

	# augment results with important arguments
	results_dict.update(
		{
			'task_name': task_name,
			'data_frac': DATA_FRAC
		}
	)

	# save results as json file
	path_analysis_root = os.path.join(cfg['root_path_results'], 'analysis')
	Path(path_analysis_root).mkdir(parents=True, exist_ok=True)
	filename = f'analysis_results_{task_name}.json'
	path_analysis_results = os.path.join(path_analysis_root, filename)

	with open(path_analysis_results, 'w') as filesave:
		json.dump(results_dict, filesave)
	

	
if __name__ == '__main__':
	main()