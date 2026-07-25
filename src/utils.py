import yaml


def parse_config(path_to_config='config.yml'):
	"""
	"""

	with open(path_to_config) as yaml_file:
		config_dict = yaml.safe_load(yaml_file)

	return config_dict


