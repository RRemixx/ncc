"""Paper driver: base predictions -> NCC -> results pkl and model weights."""
import logging
import random
from dataclasses import replace

import numpy as np
import torch

from ncc.model import NCCModel
from ncc.online import calibrate
from ncc.series import build_series
from ncc.train import fit
from ncc.experiments import paths
from ncc.experiments.files import load_pickle, save_pickle

logger = logging.getLogger(__name__)


def run_experiment(exp, seed, device=None):
    """Trains NCC on the steps before the test steps, then calibrates every step.

    device overrides exp.ncc.device for this run. Writes results/<exp_id>_<seed>.pkl (see README) and
    models/<exp_id>_<seed>.pt (the final weights), and returns the path of the results file.
    """
    cfg = exp.ncc if device is None else replace(exp.ncc, device=device)
    pred = load_pickle(paths.base_pred(exp.base_pred_id))['base_pred']
    locations = list(pred[0][0].keys())
    first_test = len(pred) - exp.test_steps
    train_end = first_test - cfg.val_steps

    _seed_everything(seed)
    data, scaler = build_series(pred, exp.horizon, train_steps=train_end)
    model = NCCModel(cfg.model, len(cfg.alphas), cfg.window_size)
    schedule = cfg.train.epochs if cfg.train.stages is None else tuple(stage.epochs for stage in cfg.train.stages)
    logger.info('training %s (seed %d) on steps [0, %d)', exp.exp_id, seed, train_end)
    fit(model, data, cfg, locations, exp.horizon, train_end=train_end, walk_end=first_test, schedule=schedule)

    # Reseed and rebuild the model before calibrating. This reproduces the random number sequence of
    # the paper's code, which trained and calibrated in two separate processes.
    trained_state = model.state_dict()
    _seed_everything(seed)
    model = NCCModel(cfg.model, len(cfg.alphas), cfg.window_size)
    model.load_state_dict(trained_state)
    logger.info('calibrating %s (seed %d), test steps from %d', exp.exp_id, seed, first_test)
    results = calibrate(model, data, scaler, cfg, locations, exp.horizon, first_test, exp.max_steps)

    results_path = paths.results(exp.exp_id, seed)
    save_pickle(results_path, results)
    weights_path = paths.weights(exp.exp_id, seed)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    logger.info('results saved to %s', results_path)
    return results_path


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
