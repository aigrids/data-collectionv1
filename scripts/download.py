""" Downloads all datasets

Example usage:

	$ python scripts/download.py

"""
import sys
from aidotgrids import load

sys.path.add('src')
import utils



config = utils.parse_config('config.yml')

ds = 