from types import SimpleNamespace
import yaml
import wandb
import numpy as np
import torch
import inspect


def load_config(path):
    config = SimpleNamespace()
    with open(path) as f:
        # Read the yaml, convert to a SimpleNamespace and return
        config.__dict__.update(yaml.load(f))
    return config


def config_diff(config, template):
    diff_keys = template.__dict__.keys() - config.__dict__.keys()
    return SimpleNamespace(**{k: template.__dict__[k] for k in diff_keys})


def get_path_to_rlexp():
    import os
    return os.getcwd()


def initialize_wandb(project, tags, path, entity='salesforce'):
    return SimpleNamespace(wandb=wandb.init(project=project, tags=tags, entity=entity, dir=path + "/wandb"))


def set_seeds(seed):
    # Set all the seeds
    np.random.seed(seed)
    torch.manual_seed(seed)


def dump_args(func):
    # Taken from
    # https://stackoverflow.com/questions/6200270/decorator-to-print-function-call-details-parameters-names-and-effective-values
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ', '.join('{} = {!r}'.format(*item) for item in func_args.items())
        print("-----------------------------------------------------------")
        print(f'{func.__module__}.{func.__qualname__} ( {func_args_str} )')
        return func(*args, **kwargs)

    return wrapper
