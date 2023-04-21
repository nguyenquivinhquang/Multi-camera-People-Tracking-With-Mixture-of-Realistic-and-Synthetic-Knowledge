"""
    https://github.com/nhtlongcs/AIC2022-VER/blob/main/opt.py
"""

from typing import Optional, Union
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import yaml
import json


def load_yaml(path):
    with open(path, "rt") as f:
        return yaml.safe_load(f)


class Config(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, yaml_path):
        super(Config, self).__init__()

        config = load_yaml(yaml_path)
        super(Config, self).update(config)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def save_yaml(self, path):
        print(f"Saving config to {path}...")
        with open(path, "w") as f:
            yaml.dump(dict(self), f, default_flow_style=False, sort_keys=False)

    def update_config(self, cfg):
        self = deep_update(dict(self), cfg)
        return dict(self)

    @classmethod
    def load_yaml(cls, path):
        print(f"Loading config from {path}...")
        return cls(path)

    def __repr__(self) -> str:
        return str(json.dumps(dict(self), sort_keys=False, indent=4))


def deep_update(mapping: dict, *updating_mappings: dict()) -> dict():
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v

    return updated_mapping


class Opts(ArgumentParser):
    def __init__(self, cfg: Optional[Union[str, dict]] = None):
        super(Opts, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        if isinstance(cfg, dict):
            self.config = cfg
        else:
            self.config = Config(cfg)
        self.add_argument(
            "-c", "--config", default="DEFAULT", help="configuration file to use"
        )
        self.add_argument(
            "-o", "--opt", nargs="+", help="override configuration options"
        )

    def parse_args(self, argv=None):
        args = super(Opts, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        if args.config is not "DEFAULT":
            _config = Config(args.config)
            self.config = deep_update(self.config, _config)
        # print(self.config)
        config = self.override(self.config, args.opt)
        print(self.config)
        return config

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = v
        return config

    def override(self, global_config, overriden):
        """
        Merge config into global config.
        Args:
            config (dict): Config to be merged.
        Returns: global config
        """
        print("Overriding configurating")
        for key, value in overriden.items():
            if "." not in key:
                if isinstance(value, dict) and key in global_config:
                    global_config[key].update(value)
                else:
                    if key in global_config.keys():
                        global_config[key] = value
                    else:
                        print(f"'{key}' not found in config, create a new one")
                    global_config[key] = value
            else:
                sub_keys = key.split(".")
                assert (
                    sub_keys[0] in global_config
                ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                    global_config.keys(), sub_keys[0]
                )
                cur = global_config[sub_keys[0]]
                for idx, sub_key in enumerate(sub_keys[1:]):
                    if idx == len(sub_keys) - 2:
                        if sub_key in cur.keys():
                            cur[sub_key] = value
                        else:
                            print(f"'{key}' not found in config")
                    else:
                        cur = cur[sub_key]
        return global_config
