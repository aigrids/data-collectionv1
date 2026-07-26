""" Contains functions used throughout repository, removing need for duplicates.

"""
import yaml


def parse_config(path_to_config):
	""" Load configuration .yaml file as dictionary. 
	"""

	with open(path_to_config) as yaml_file:
		config_dict = yaml.safe_load(yaml_file)

	return config_dict


