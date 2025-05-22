"""All utilities for configurations."""

import json
from dataclasses import asdict
from pprint import pprint
from types import SimpleNamespace

import yaml


def load_config(path: str):
    """
    Loads the configuration yaml and parses it into an object with dot access.
    """
    with open(path, encoding="utf-8") as stream:
        # Load dict
        config_dict = yaml.safe_load(stream)

        # Convert to namespace (access via config.data etc)
        config = json.loads(json.dumps(config_dict), object_hook=load_object)
    return config, config_dict


def save_config(config, path: str):
    """
    Save the configuration yaml.
    """
    print(f"Saving config to {path}")
    with open(path, "w", encoding="utf-8") as stream:
        # Load dict
        yaml.dump(
            json.loads(json.dumps(config, default=lambda s: vars(s))),
            default_flow_style=False,
            stream=stream,
        )


def print_config(config):
    print(
        yaml.dump(
            json.loads(json.dumps(config, default=lambda s: vars(s))),
            default_flow_style=False,
        )
    )
