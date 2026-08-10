""" Downloads all datasets

Example usage:

	$ python scripts/download.py

"""
from aidotgrids import load

import utils

MAX_WORKERS = 128
MAX_WORKERS_DOWNLOAD = 64
DATA_FRAC = 0.01
MAP_TASK_TO_SAMPLE_SUBTASK = {
	'OPFData': 'train_small_test_medium',
	'PowerGraph': 'cascading_failure_binary',
	'SolarCube': 'odd_time_area',
	'BuildingElectricity': 'odd_time_buildings92',
	'WindFarm': 'odd_time_predict48h'
}


def main():
	config = utils.parse_config('config.yml')
	root = config['root_path_datasets']

	for task in config['list_avail_tasknames']:
		subtask = MAP_TASK_TO_SAMPLE_SUBTASK[task]

		ds = load.load_task(
			task_name=task,
			subtask_name=subtask,
			root_path=root,
			max_workers=MAX_WORKERS,
			max_workers_download=MAX_WORKERS_DOWNLOAD,
			data_frac=DATA_FRAC
		)


if __name__ == '__main__':
	main()
