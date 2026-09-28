"""Experiment configs: configs/experiments/<exp_id>.yaml merged onto configs/default.yaml."""
from dataclasses import dataclass

from ncc.config import NCCConfig
from ncc.experiments import paths
from ncc.experiments.files import load_pickle, load_yaml

_REQUIRED = ('base_pred_id', 'horizon', 'test_steps', 'max_steps', 'seed', 'seeds')


@dataclass(frozen=True)
class Experiment:
    exp_id: str           # name of configs/experiments/<exp_id>.yaml
    base_pred_id: int     # base predictions: results/base_pred/saved_pred_<base_pred_id>.pickle
    horizon: int          # steps ahead, counted from 1
    test_steps: int       # the last test_steps time steps are the test steps
    max_steps: int        # the run stops after step max_steps
    seed: int             # default seed of `run` and `evaluate`
    seeds: tuple          # seeds evaluated by `evaluate -u`
    ncc: NCCConfig        # the method's config


def load_experiment(exp_id):
    """Loads an experiment. Nested mappings are merged recursively onto default.yaml; lists are replaced.

    Raises ValueError on an unknown or missing key.
    """
    d = _merge(load_yaml(paths.configs() / 'default.yaml') or {},
               load_yaml(paths.configs() / 'experiments' / f'{exp_id}.yaml') or {})
    ncc = NCCConfig.from_dict(d.pop('ncc', None) or {})
    unknown = sorted(set(d) - set(_REQUIRED))
    missing = [key for key in _REQUIRED if d.get(key) is None]
    if unknown or missing:
        raise ValueError(f'experiment {exp_id}: unknown key(s) {unknown}, missing key(s) {missing}')
    d['seeds'] = tuple(d['seeds'])
    return Experiment(exp_id=str(exp_id), ncc=ncc, **d)


def dataset_name(exp):
    """Dataset name stored in the experiment's base prediction file."""
    return load_pickle(paths.base_pred(exp.base_pred_id))['params']['dataset']


def _merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
