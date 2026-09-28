"""Training of the NCC model on replays of the visible history: fit(), the loss log and early stopping."""
import logging
from dataclasses import replace

import numpy as np
import torch

from ncc.history import CalibrationHistory
from ncc.losses import NCCLoss

logger = logging.getLogger(__name__)

LR_STEP_EPOCHS = 50     # StepLR decays the learning rate by gamma every this many epochs
PATIENCE_EPOCHS = 50    # early stopping patience


def fit(model, store, scaler, locations, origins, t, epochs, cfg, np_rng):
    """Trains model in place at step t and returns the LossLog of all epochs.

    Uses only what the store holds, i.e. what is visible at t. Each epoch replays a fresh history:
    gradient steps over `origins` (the scored training origins, ascending), then a gradient-free walk
    over the origins after them up to t - 1. The misses of the replay come from its own q-hats.
    epochs: the epochs of the single run (int), or of each stage when cfg.train.stages is set.
    Every run (each stage) has its own optimizer, learning rate scheduler and early stopping.
    np_rng: the np.random.RandomState that shuffles the locations of each batch.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    walk = range(origins[-1] + 1, t)

    def new_replay():
        return CalibrationHistory(store, scaler, locations, cfg, np_rng)

    def targets(batch, s):
        """Normalized scores of origin s (B x 1), in the batch's order."""
        scores = np.array([[store.normalized_score(r, s)] for r in batch.locations], dtype=np.float32)
        return torch.from_numpy(scores).to(device)

    log = LossLog()
    for run_epochs, weights in _runs(cfg, epochs):
        loss_fn = NCCLoss(weights, cfg.train.K, cfg.alphas, device)
        _fit_run(model, new_replay, targets, locations, origins, walk, int(run_epochs), loss_fn, cfg, log)
    return log


def _runs(cfg, epochs):
    """(epochs, loss weights) of each run. A stage overrides the quantile, coverage and efficiency weights."""
    if cfg.train.stages is None:
        return [(epochs, cfg.train.weights)]
    return [
        (stage_epochs, replace(cfg.train.weights, quantile=q, coverage=c, efficiency=e))
        for stage_epochs, (q, c, e) in zip(epochs, (stage.weights for stage in cfg.train.stages))
    ]


def _fit_run(model, new_replay, targets, locations, origins, walk, epochs, loss_fn, cfg, log):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_EPOCHS, gamma=cfg.train.gamma)
    patience = Patience(PATIENCE_EPOCHS)
    for _ in range(epochs):
        replay = new_replay()
        model.train()
        for s in origins:
            batch = replay.window(s, locations)
            q_hats = _model_output(model, batch)
            loss, parts = loss_fn(q_hats, batch.err_seq, batch.input_alphas, targets(batch, s))
            log.add(parts)
            optimizer.zero_grad()
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
            except RuntimeError:
                logger.warning('Infinite loss, skipped!')
            replay.record(s, batch.locations, q_hats, 'train')
        # TODO(paper): the gradient-free walk only advances the replay; its outputs are unused,
        # and early stopping reads the training loss
        model.eval()
        with torch.no_grad():
            for s in walk:
                batch = replay.window(s, locations)
                replay.record(s, batch.locations, _model_output(model, batch), 'train')
        total = log.epoch_end()
        scheduler.step()
        if patience(total):
            break
    # TODO(paper): no best state is restored; the model keeps the weights of the last epoch


def _model_output(model, batch):
    """Normalized q-hats of a batch, from the model alone (no ACI, no TTA)."""
    return model(batch.err_seq, batch.score_seq, batch.qhat_seq, batch.input_alphas)


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
