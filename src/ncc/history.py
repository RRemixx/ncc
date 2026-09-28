"""Calibration history: the past forecasts, truths, scores and misses the model reads through a window."""
from typing import NamedTuple

import numpy as np
import torch

SOURCES = ('train', 'replay', 'online')


class Batch(NamedTuple):
    """Model inputs for one step. B = locations, W = window size, A = alphas."""
    locations: list             # location of each row (shuffled order)
    err_seq: torch.Tensor       # (B x W x A): past misses
    score_seq: torch.Tensor     # (B x W x 3): y_hat, y, score (normalized)
    qhat_seq: torch.Tensor      # (B x W x A): zeros
    input_alphas: torch.Tensor  # (B x A)
    q_adj: torch.Tensor         # (B x A): accumulated ACI correction
    window_origins: np.ndarray  # (B x W): origin of each window entry, -1 for a placeholder


class _Entry:
    """One issued (or replayed) q-hat row of a location. Completed once the store has its feedback."""
    __slots__ = ('origin', 'q_hat', 'y_hat', 'source', 'features', 'miss')

    def __init__(self, origin, q_hat, y_hat, source):
        self.origin = origin
        self.q_hat = q_hat          # (A,) float32, normalized
        self.y_hat = y_hat          # float32, normalized center
        self.source = source        # 'train', 'replay' or 'online'
        self.features = None        # (3,) float32: y_hat, y, score (normalized), once completed
        self.miss = None            # (A,) float32: 1 if missed, once completed


class CalibrationHistory:
    """Per-location entries ordered by origin, placed when their origin's feedback arrives.

    The window at step t of a location holds the window_size completed entries with the largest
    origins among those whose feedback arrived at or before t, sorted by origin. Placeholders fill the
    front while fewer exist: miss = 1 (or alpha), and (y_hat, y, score) = 0.
    A miss is score_n > q_hat_n, from the entry's own q-hat. Only for an 'online' entry, an external
    miss given with the feedback replaces it, per alpha.
    """

    def __init__(self, store, scaler, locations, cfg, rng):
        """store: SeriesStore; scaler: fitted Scaler; cfg: NCCConfig; rng: np.random.RandomState of the shuffles."""
        self.store = store
        self.scaler = scaler
        self.rng = rng
        self.alphas = cfg.alphas
        self.window_size = cfg.window_size
        self.device = torch.device(cfg.device)
        placeholder = np.ones(len(cfg.alphas), dtype=np.float32)
        if cfg.init_err_with_alpha:
            placeholder = placeholder * np.asarray(cfg.alphas, dtype=np.float32)
        self._placeholder_miss = placeholder
        self._entries = {location: [] for location in locations}

    def record(self, origin, locations, q_hats, source):
        """Creates the entries of origin for locations; q_hats (B x A, normalized) has one row per location.

        Origins must be recorded in ascending order per location.
        """
        assert source in SOURCES
        q_hats = q_hats.detach().cpu().numpy() if torch.is_tensor(q_hats) else np.asarray(q_hats)
        for q_hat, location in zip(q_hats, locations):
            y_hat = np.float32(self.scaler.transform(location, self.store.center(location, origin)))
            self._entries[location].append(_Entry(origin, q_hat.astype(np.float32), y_hat, source))

    def window(self, t, locations, q_adj=None):
        """Batch of step t: locations in a new random order, each with its window at t.

        q_adj: {location: tensor (A,)}, the ACI state; zeros when None.
        """
        order = np.arange(len(locations))
        self.rng.shuffle(order)
        batch_locations = [locations[i] for i in order]
        num_alphas, w = len(self.alphas), self.window_size
        err_seq = np.empty((len(order), w, num_alphas), dtype=np.float32)
        score_seq = np.zeros((len(order), w, 3), dtype=np.float32)
        window_origins = np.full((len(order), w), -1, dtype=np.int64)
        for b, location in enumerate(batch_locations):
            entries = self._latest(location, t)
            pad = w - len(entries)
            err_seq[b, :pad] = self._placeholder_miss
            for i, entry in enumerate(entries, start=pad):
                err_seq[b, i] = entry.miss
                score_seq[b, i] = entry.features
                window_origins[b, i] = entry.origin
        # TODO(paper): the q-hat history is never written, so the q-hat encoder always sees zeros
        qhat_seq = torch.zeros((len(order), w, num_alphas))
        input_alphas = torch.tensor(self.alphas)[None, :].repeat(len(order), 1)
        if q_adj is None:
            q_adj_rows = torch.zeros((len(order), num_alphas))
        else:
            q_adj_rows = torch.stack([q_adj[location] for location in batch_locations])
        return Batch(batch_locations, torch.from_numpy(err_seq).to(self.device),
                     torch.from_numpy(score_seq).to(self.device), qhat_seq.to(self.device),
                     input_alphas.to(self.device), q_adj_rows.to(self.device), window_origins)

    def _latest(self, location, t):
        """The window_size latest completed entries of location with feedback arrived at or before t, by origin."""
        latest = []
        for entry in reversed(self._entries[location]):
            if len(latest) == self.window_size:
                break
            if self._complete(location, entry) and self.store.arrived(location, entry.origin) <= t:
                latest.append(entry)
        latest.reverse()
        return latest

    def _complete(self, location, entry):
        """Completes entry from the store if its feedback has arrived; returns whether it is complete."""
        if entry.features is not None:
            return True
        if not self.store.has_feedback(location, entry.origin):
            return False
        origin = entry.origin
        y = np.float32(self.scaler.transform(location, self.store.truth(location, origin)))
        score = self.store.normalized_score(location, origin)
        miss = (score > entry.q_hat).astype(np.float32)
        external_miss = self.store.external_miss(location, origin)
        if entry.source == 'online' and external_miss is not None:
            for i, external in enumerate(external_miss):
                if external is not None:
                    miss[i] = float(bool(external))
        entry.features = np.array([entry.y_hat, y, score], dtype=np.float32)
        entry.miss = miss
        return True
