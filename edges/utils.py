import yaml


def parse_config(config):
    # Directly return dicts
    if isinstance(config, dict):
        return config

    # Assume strings are filenames for yaml files, and parse them
    if isinstance(config, str):
        with open(config, "r") as f:
            try:
                config = yaml.load(f)
                return config
            except yaml.YAMLError as exc:
                print(exc)

    # TypeError
    raise TypeError('config must be dict or filename to a yaml file.')
