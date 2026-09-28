"""File locations, relative to the repository root.

Every function reads the module-level ROOT when called, so tests can monkeypatch it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root (with an editable install)


def base_pred(base_pred_id):
    """Saved base forecaster predictions."""
    return ROOT / 'results' / 'base_pred' / f'saved_pred_{base_pred_id}.pickle'


def results(exp_id, seed):
    """NCC results of one run."""
    return ROOT / 'results' / f'{exp_id}_{seed}.pkl'


def weights(exp_id, seed):
    """Final model weights of one run."""
    return ROOT / 'models' / f'{exp_id}_{seed}.pt'


def csv_dir():
    """Directory of the metric CSVs."""
    return ROOT / 'results' / 'csvs'


def clip_idx(dataset):
    """Time steps of the test period that `evaluate -c` keeps."""
    return ROOT / 'data' / f'{dataset}_clip_idx.pkl'


def configs():
    """Directory with default.yaml and experiments/<exp_id>.yaml."""
    return ROOT / 'configs'
