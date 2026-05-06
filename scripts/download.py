""" Downloads all datasets

Example usage:

	$ python scripts/download.py

"""
import sys
from aidotgrids import load

sys.path.add('src')
import utils

MAX_WORKERS = 64
MAX_WORKERS_DOWNLOAD = 32


config = utils.parse_config('config.yml')

print(config)

def main():

	ds = load.load_task(
		task_name='WindFarm',
		subtask_name='odd_time_predict48h',
		root_path='/hpcwork/p0027791/AI-grids',
		max_workers=MAX_WORKERS,
		max_workers_download=MAX_WORKERS_DOWNLOAD,
		data_frac=0.01
	)


if __name__ == '__main__':
	pass
	#main()
