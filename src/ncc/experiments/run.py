"""Paper driver: base predictions -> NCC, one step at a time -> results pkl and model weights."""
import logging
from dataclasses import replace

import numpy as np
import torch

from ncc.online import NCC, Feedback
from ncc.experiments import paths
from ncc.experiments.files import load_pickle, save_pickle

logger = logging.getLogger(__name__)


def run_experiment(exp, seed, device=None):
    """Runs NCC step by step over the base predictions and saves the forecasts of the test steps.

    The truth of step s becomes visible exp.horizon steps later. NCC warms up (trains) at the first
    test step: n_warmup is derived from the number of steps, which is fine for reproducing the paper
    but not prefix-stable. device overrides exp.ncc.device for this run.
    Writes results/<exp_id>_<seed>.pkl (see README) and models/<exp_id>_<seed>.pt (the final weights),
    and returns the path of the results file. When no model was trained (max_steps < the first test
    step), no weights file is written and the results hold empty lists.
    """
    cfg = exp.ncc if device is None else replace(exp.ncc, device=device)
    pred = load_pickle(paths.base_pred(exp.base_pred_id))['base_pred']
    locations = list(pred[0][0].keys())
    h = exp.horizon
    # horizon is 1-based, the prediction arrays are indexed from 0
    yhat = [{r: float(predictions[r][h - 1]) for r in locations} for predictions, _ in pred]
    y = [{r: float(infos[r][0][h - 1]) for r in locations} for _, infos in pred]
    num_steps = len(pred)
    first_test = num_steps - exp.test_steps
    ncc = NCC(replace(cfg, n_warmup=first_test - h + 1), seed)

    logger.info('running %s (seed %d), test steps from %d', exp.exp_id, seed, first_test)
    results = {r: {h: []} for r in locations}
    for t in range(min(num_steps, exp.max_steps + 1)):
        s = t - h
        feedback = [Feedback(s, r, y[s][r], abs(y[s][r] - yhat[s][r])) for r in locations] if s >= 0 else []
        q_hats = ncc.step(t, yhat[t], feedback)
        if t >= first_test:
            for r in locations:
                results[r][h].append((q_hats[r], np.array([yhat[t][r]]), np.array([y[t][r]])))

    results_path = paths.results(exp.exp_id, seed)
    save_pickle(results_path, results)
    if ncc.model is not None:
        weights_path = paths.weights(exp.exp_id, seed)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ncc.model.state_dict(), weights_path)
    else:
        logger.warning('no model was trained (the run stopped before the first test step); no weights saved')
    logger.info('results saved to %s', results_path)
    return results_path
