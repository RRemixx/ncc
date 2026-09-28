"""Sequential calibration, one step at a time: the NCC method object, with retraining, ACI and test-time adaptation (TTA)."""
import logging
import math
import numbers
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Sequence as SequenceABC
from typing import Mapping, Sequence

import numpy as np
import torch

from ncc.config import NCCConfig
from ncc.history import CalibrationHistory
from ncc.losses import MonotonicityLoss, monotonicity_fraction
from ncc.model import NCCModel
from ncc.series import Scaler, SeriesStore
from ncc.train import fit

logger = logging.getLogger(__name__)

# TTA runs when more than this fraction of the batch has q-hats that increase with alpha
TTA_MONOTONICITY_THRESHOLD = 0.05
TTA_MAX_STEPS = 200
TTA_CHECK_EVERY = 10    # TTA checks every this many steps whether it can stop


@dataclass(frozen=True)
class Feedback:
    """The grade of one task (origin, location), handed to NCC.step when it becomes visible."""
    origin: int                     # step index of the graded origin
    location: str
    truth: float                    # first-reported truth, target units, finite
    score: float                    # nonconformity score of the task (one per task), finite, >= 0, in target
                                    # units (NCC normalizes it by the truth's std, so it must scale with the
                                    # target, e.g. |y - y_hat|)
    external_miss: tuple[bool | None, ...] | None = None
    # optional, per alpha: whether the interval that was actually delivered missed (bool). None, or None for
    # one alpha -> NCC's own issued q-hat decides the miss for that alpha. The alpha = 1 entry is normally None.


class NCC:
    """NCC as a stateful method: called once per forecast origin, in ascending order.

    The caller decides what is visible at each step and hands it over; NCC never sees the future.
    The output at step t depends only on the inputs of steps <= t (prefix stability). On CPU the
    object's random generators are its own, so several objects in one process stay independent;
    use them from one thread.

    Attributes read by tests: fit_logs (one LossLog per fit: warm-up and each retrain), warmup_step.
    """

    def __init__(self, cfg: NCCConfig, seed: int):
        """cfg: NCCConfig with n_warmup set; seed: seed of the model and training.

        Raises ValueError if cfg.n_warmup is None or < 1.
        """
        if cfg.n_warmup is None or cfg.n_warmup < 1:
            raise ValueError(f"cfg.n_warmup must be an int >= 1, got {cfg.n_warmup}")
        self.cfg = cfg
        self.scaler = Scaler()
        self.store = SeriesStore(self.scaler)
        self.locations = None           # fixed by the first call's centers
        self.model = None               # built at warm-up
        self.fit_logs = []
        self.history = None             # the online history, built at warm-up
        self.q_adj = {}                 # location -> tensor (A,): the ACI state
        self.issued_steps = 0
        self.warmup_step = None
        self.np_rng = np.random.RandomState(seed)
        self.torch_state = torch.Generator().manual_seed(seed).get_state()
        _seed_device_generator(cfg.device, seed)
        self._last_t = -1
        self._last_feedback_origin = None

    def step(self, t: int, centers: Mapping[str, float], feedback: Sequence[Feedback]) -> dict[str, np.ndarray]:
        """Advance to step t (0 on the first call, then previous t + 1).

        centers: finite base forecast (point, else the median) of every location, at every step.
            The first call fixes the location set and its order; later calls must have exactly those keys.
        feedback: the grades that became visible at t: empty until the first step that brings any, then
            exactly one row per location for exactly one origin, one more than the previous step's (the
            first is 0), and before t. Truths and scores finite, scores >= 0, external_miss None or one
            entry per alpha.
        Returns {location: q-hat of shape (len(alphas),), float64, target units} for every location;
        {} while warming up. q-hat is returned as NCC produced it: not clamped at 0 and possibly
        non-monotone in alpha (the caller builds, clamps and repairs the intervals).
        Raises ValueError on malformed input, before changing any state. No data condition raises
        anything else; errors from torch itself (e.g. out of memory, or inside TTA) propagate unchanged.
        """
        locations = self.locations if self.locations is not None else list(centers)
        self._check_step(t, centers, locations)
        self._check_feedback(t, feedback, locations)

        self.locations = locations
        self._last_t = t
        self.store.add_centers(t, centers)
        for fb in feedback:
            self.store.add_truth(fb, arrived=t)
        if feedback:
            self._last_feedback_origin = feedback[0].origin

        with self._own_torch_rng():
            if self.model is None:
                if self.store.num_scored_origins() < self.cfg.n_warmup or not self._train_origins(t):
                    return {}
                self._warm_up(t)
            issued = self._infer(t, 'online')
            self.issued_steps += 1
            if self.issued_steps % self.cfg.retrain.period == 0:
                # the online history is not rebuilt after a retrain
                logger.info('retrain started at step %d', t)
                self._fit(t, self._retrain_epochs())
                logger.info('retrain finished')
        return issued

    def _train_origins(self, t):
        """The scored origins s < t - val_steps, ascending."""
        return [s for s in self.store.scored_origins() if s < t - self.cfg.val_steps]

    def _warm_up(self, t):
        """Fits the scalers, builds and trains the model, then replays origins 0 .. t-1 to build the online history."""
        cfg = self.cfg
        self.warmup_step = t
        origins = self._train_origins(t)
        for location in self.locations:
            self.scaler.fit(location, np.array([self.store.truth(location, s) for s in origins]))
        self.model = NCCModel(cfg.model, len(cfg.alphas), cfg.window_size).to(torch.device(cfg.device))
        logger.info('warm-up at step %d: training on origins %d .. %d', t, origins[0], origins[-1])
        initial_epochs = cfg.train.epochs if cfg.train.stages is None else tuple(stage.epochs for stage in cfg.train.stages)
        self._fit(t, initial_epochs)
        # replay as if the trained model had run online from the start; misses from its own q-hats
        self.history = CalibrationHistory(self.store, self.scaler, self.locations, cfg, self.np_rng)
        self.q_adj = {location: torch.zeros(len(cfg.alphas)) for location in self.locations}
        for s in range(t):
            self._infer(s, 'replay')

    def _fit(self, t, epochs):
        """Trains the model at step t on the training origins, and keeps the loss log."""
        log = fit(self.model, self.store, self.scaler, self.locations, self._train_origins(t), t, epochs,
                  self.cfg, self.np_rng)
        self.fit_logs.append(log)

    def _retrain_epochs(self):
        return self.cfg.retrain.epochs if self.cfg.train.stages is None else self.cfg.retrain.stage_epochs

    def _infer(self, t, source):
        """q-hats of step t for every location (with TTA and ACI if enabled), recorded in the online history.

        Returns {location: q-hat (A,) float64, target units}, in the location order.
        """
        cfg = self.cfg
        self.model.eval()
        batch = self.history.window(t, self.locations, self.q_adj)
        if cfg.tta.enabled:
            q_hats, q_adj = _tta_predict(self.model, batch, t, cfg)
        else:
            # TODO(paper): ACI only runs inside TTA
            with torch.no_grad():
                q_hats = self.model(batch.err_seq, batch.score_seq, batch.qhat_seq, batch.input_alphas)
            q_adj = batch.q_adj
        self.history.record(t, batch.locations, q_hats, source)
        q_hats = q_hats.detach().cpu().numpy()
        q_adj = q_adj.detach().cpu()
        issued = {}
        for r, location in enumerate(batch.locations):
            self.q_adj[location] = q_adj[r]
            issued[location] = self.scaler.inverse(location, q_hats[r].astype(np.float64), is_score=True)
        return {location: issued[location] for location in self.locations}

    @contextmanager
    def _own_torch_rng(self):
        """Runs the block on this object's CPU torch generator state, leaving the global one untouched."""
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.torch_state)
            yield
            self.torch_state = torch.get_rng_state()

    def _check_step(self, t, centers, locations):
        """Raises ValueError if t is not the next step, or the centers are not one finite value per location."""
        if not _is_integer(t) or t != self._last_t + 1:
            raise ValueError(f"step t={t}: expected t={self._last_t + 1}")
        if not centers:
            raise ValueError(f"step {t}: centers is empty")
        known = set(locations)
        missing = [r for r in locations if r not in centers]
        extra = [r for r in centers if r not in known]
        if missing or extra:
            raise ValueError(f"step {t}: centers must have the locations of the first step; "
                             f"missing {missing}, extra {extra}")
        for location, center in centers.items():
            if not _is_finite(center):
                raise ValueError(f"step {t}: center of location {location!r} is not a finite number: {center!r}")

    def _check_feedback(self, t, feedback, locations):
        """Raises ValueError unless feedback is empty (before the first feedback) or covers the next origin completely."""
        expected = 0 if self._last_feedback_origin is None else self._last_feedback_origin + 1
        if not feedback:
            if self._last_feedback_origin is not None:
                raise ValueError(f"step {t}: no feedback, expected feedback for origin {expected}")
            return
        for fb in feedback:
            if not isinstance(fb, Feedback):
                raise ValueError(f"step {t}: feedback row is not a Feedback: {fb!r}")
        origins = sorted({fb.origin for fb in feedback})
        if origins != [expected]:
            raise ValueError(f"step {t}: feedback covers origin(s) {origins}, expected exactly origin {expected}")
        if expected >= t:
            raise ValueError(f"step {t}: feedback for origin {expected} is not before step {t}")
        known, seen = set(locations), set()
        for fb in feedback:
            if fb.location not in known:
                raise ValueError(f"step {t}: feedback for unknown location {fb.location!r}")
            if fb.location in seen:
                raise ValueError(f"step {t}: duplicate feedback for location {fb.location!r}")
            seen.add(fb.location)
            if not _is_finite(fb.truth):
                raise ValueError(f"step {t}: truth of location {fb.location!r} is not a finite number: {fb.truth!r}")
            if not _is_finite(fb.score) or fb.score < 0:
                raise ValueError(f"step {t}: score of location {fb.location!r} must be finite and >= 0, "
                                 f"got {fb.score!r}")
            if fb.external_miss is not None and not _is_sequence(fb.external_miss):
                raise ValueError(f"step {t}: external_miss of location {fb.location!r} must be None or a "
                                 f"sequence, got {fb.external_miss!r}")
            if fb.external_miss is not None and len(fb.external_miss) != len(self.cfg.alphas):
                raise ValueError(f"step {t}: external_miss of location {fb.location!r} has "
                                 f"{len(fb.external_miss)} entries, expected {len(self.cfg.alphas)}")
        missing = [r for r in locations if r not in seen]
        if missing:
            raise ValueError(f"step {t}: feedback for origin {expected} is missing location(s) {missing}")


def _is_finite(value):
    """A finite real number (not a bool)."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _is_integer(value):
    """An integer (not a bool)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_sequence(value):
    """A sequence such as a tuple or a list (not a string)."""
    return isinstance(value, SequenceABC) and not isinstance(value, str)


def _seed_device_generator(device, seed):
    """Seeds the process-global generator of an accelerator device (dropout draws from it in training)."""
    kind = torch.device(device).type
    if kind == 'cuda':
        torch.cuda.manual_seed(seed)
    elif kind == 'mps':
        torch.mps.manual_seed(seed)


def _tta_predict(model, batch, t, cfg):
    """q-hats with ACI (after two windows), plus the TTA head's adjustment when they are not monotone.

    The TTA head is fitted on the monotonicity loss plus reg_factor * the RMS of its output, with all
    other weights frozen. Returns (q_hats, q_adj).
    """
    mono_loss = MonotonicityLoss()
    with frozen_except(model, model.tta_head):
        optimizer = torch.optim.Adam(model.tta_head.parameters(), lr=cfg.tta.lr)
        # every other weight is frozen and dropout is off in eval mode, so the embedding is computed once
        emb = model.embed(batch.err_seq, batch.score_seq, batch.qhat_seq, batch.input_alphas)
        q_hats = model.quantile_head(emb)
        q_adj = batch.q_adj
        if t > 2 * cfg.window_size:
            q_hats, q_adj = aci_correct(q_hats, batch.err_seq, batch.input_alphas, cfg.tta.aci_lr, batch.q_adj)
        if monotonicity_fraction(q_hats) <= TTA_MONOTONICITY_THRESHOLD:
            return q_hats, q_adj
        base_q_hats = q_hats
        for step in range(1, TTA_MAX_STEPS + 1):
            hs = model.tta_head(emb)
            q_hats = base_q_hats + hs
            if step % TTA_CHECK_EVERY == 0 and monotonicity_fraction(q_hats) < TTA_MONOTONICITY_THRESHOLD:
                break
            optimizer.zero_grad()
            loss_m = mono_loss(q_hats, batch.input_alphas)
            loss_reg = hs.pow(2).mean().sqrt()
            tta_loss = loss_m + loss_reg * cfg.tta.reg_factor
            tta_loss.backward()
            optimizer.step()
    return q_hats, q_adj


def aci_correct(q, err_seq, alphas, lr, q_adj):
    """ACI-style correction: shift q-hats by the running gap between observed and target miss rates.

    Args:
        q (torch.Tensor): model output (batch size x num alphas)
        err_seq (torch.Tensor): misses in the window (batch size x window size x num alphas)
        alphas (torch.Tensor): (batch size x num alphas)
        lr (float): step size of the correction
        q_adj (torch.Tensor): accumulated correction (batch size x num alphas)

    Returns:
        (corrected q, updated q_adj)
    """
    avg_err = torch.mean(err_seq, dim=1)
    new_q_adj = q_adj + lr * (avg_err - alphas)
    adjusted_q = q + new_q_adj
    return adjusted_q, new_q_adj


@contextmanager
def frozen_except(model, module):
    """Freezes every parameter of model except those of module; unfreezes all of them on exit."""
    for param in model.parameters():
        param.requires_grad = False
    for param in module.parameters():
        param.requires_grad = True
    try:
        yield
    finally:
        for param in model.parameters():
            param.requires_grad = True
