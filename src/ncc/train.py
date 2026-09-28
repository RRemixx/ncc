"""Training of the NCC model on the calibration history: fit(), the loss log and early stopping."""
import logging
from dataclasses import replace

import numpy as np
import torch

from ncc.history import CalibrationHistory
from ncc.losses import NCCLoss

logger = logging.getLogger(__name__)

LR_STEP_EPOCHS = 50     # StepLR decays the learning rate by gamma every this many epochs
PATIENCE_EPOCHS = 50    # early stopping patience


def fit(model, data, cfg, regions, horizon, train_end, walk_end, schedule):
    """Trains model in place and returns the LossLog of all epochs.

    Each epoch replays the history from scratch: gradient steps over [0, train_end), then a
    gradient-free walk over [train_end, walk_end).
    schedule: the epochs of the single run (int), or the epochs of each stage when cfg.train.stages is set.
    Every run (each stage) has its own optimizer, learning rate scheduler and early stopping.
    """
    log = LossLog()
    for epochs, weights in _runs(cfg, schedule):
        _fit_run(model, data, cfg, regions, horizon, train_end, walk_end, int(epochs), weights, log)
    return log


def _runs(cfg, schedule):
    """(epochs, loss weights) of each run. A stage overrides the quantile, coverage and efficiency weights."""
    if cfg.train.stages is None:
        return [(schedule, cfg.train.weights)]
    return [
        (epochs, replace(cfg.train.weights, quantile=q, coverage=c, efficiency=e))
        for epochs, (q, c, e) in zip(schedule, (stage.weights for stage in cfg.train.stages))
    ]


def _fit_run(model, data, cfg, regions, horizon, train_end, walk_end, epochs, weights, log):
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    loss_fn = NCCLoss(weights, cfg.train.K, cfg.alphas, device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_EPOCHS, gamma=cfg.train.gamma)
    patience = Patience(PATIENCE_EPOCHS)
    num_steps = len(data[regions[0]])
    for _ in range(epochs):
        history = CalibrationHistory(regions, cfg.alphas, cfg.window_size + num_steps + horizon - 1,
                                     cfg.init_err_with_alpha)
        model.train()
        for t in range(train_end):
            q_hats, batch = _forward_pass(model, history, data, regions, t, cfg, horizon, device)
            loss, parts = loss_fn(q_hats, batch.err_seq, batch.input_alphas, batch.scores)
            log.add(parts)
            optimizer.zero_grad()
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
            except RuntimeError:
                logger.warning('Infinite loss, skipped!')
        # TODO(paper): the gradient-free walk only advances the history; its outputs are unused,
        # and early stopping reads the training loss
        model.eval()
        for t in range(train_end, walk_end):
            _forward_pass(model, history, data, regions, t, cfg, horizon, device)
        total = log.epoch_end()
        scheduler.step()
        if patience(total):
            break
    # TODO(paper): no best state is restored; the model keeps the weights of the last epoch


def _forward_pass(model, history, data, regions, t, cfg, horizon, device):
    """Model output at step t (no ACI, no TTA), written into the history. Returns (q_hats, batch)."""
    batch = history.batch(data, regions, t, cfg.window_size, device)
    q_hats = model(batch.err_seq, batch.score_seq, batch.qhat_seq, batch.input_alphas)
    history.update(t + cfg.window_size + horizon - 1, batch, q_hats, batch.q_adj)
    return q_hats, batch


class LossLog:
    """Per-epoch means of the four loss parts and their total."""

    PARTS = ('quantile', 'coverage', 'efficiency', 'monotonicity')
    MAX_COVERAGE = 1e7  # a coverage part above this is logged as NaN

    def __init__(self):
        self.epochs = {name: [] for name in self.PARTS + ('total',)}
        self._steps = {name: [] for name in self.PARTS}

    def add(self, parts):
        """Adds the four weighted parts (tensors) of one gradient step."""
        quantile, coverage, efficiency, monotonicity = [part.detach().cpu().numpy() for part in parts]
        coverage = np.nan if coverage > self.MAX_COVERAGE else coverage
        for name, value in zip(self.PARTS, (quantile, coverage, efficiency, monotonicity)):
            self._steps[name].append(value)

    def epoch_end(self):
        """Closes the epoch and returns its total: the sum of the nanmeans of the parts."""
        total = 0
        for name in self.PARTS:
            mean = np.nanmean(self._steps[name])
            self.epochs[name].append(mean)
            total += mean
        self.epochs['total'].append(total)
        logger.info('epoch %d, total loss: %s (%s)', len(self.epochs['total']), total,
                    ', '.join(f'{name}: {self.epochs[name][-1]}' for name in self.PARTS))
        self._steps = {name: [] for name in self.PARTS}
        return total


class Patience:
    """Early stopping on the epoch total: stops after `patience` epochs in a row without improvement."""

    def __init__(self, patience):
        self.patience = patience
        self.best = None
        self.counter = 0

    def __call__(self, total):
        """Records one epoch total; returns True when training should stop.

        A NaN score compares False, so it resets the counter and becomes the best score.
        """
        score = -total
        if self.best is None:
            self.best = score
        elif score < self.best:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best = score
            self.counter = 0
        return False
