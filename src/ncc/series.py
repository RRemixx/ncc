"""Per-location data of the sequential pass: the scaler, and the store of centers and feedback."""
import numpy as np


class Scaler:
    """Standardizes y and y_hat per location. Scores (e.g. |y - y_hat|) are only rescaled."""

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, location, truths):
        """Fits the mean and std of one location on its truths (a float64 array).

        A constant location (std 0) gets std 1, so it cannot produce NaN in the shared model.
        """
        std = np.std(truths)
        self.stds[location] = std if std != 0 else 1.0
        self.means[location] = np.mean(truths)

    def transform(self, location, v):
        """v -> (v - mean) / std."""
        v = v - self.means[location]
        v = v / self.stds[location]
        return v

    def inverse(self, location, v, is_score=False):
        """v -> v * std + mean, or v * std for a score."""
        v = v * self.stds[location]
        if not is_score:
            v = v + self.means[location]
        return v


class SeriesStore:
    """Everything the caller handed over, per location and origin, in target units (float64).

    An origin's center is added at the step of that origin; its feedback (truth, score, optional
    external misses) is added later, at the step it arrives. The store does not validate its input:
    NCC.step checks it first.
    """

    def __init__(self, scaler):
        """scaler: the Scaler that normalizes the scores (fitted later, before the first normalized_score)."""
        self.scaler = scaler
        self._centers = {}      # (location, origin) -> center
        self._feedback = {}     # (location, origin) -> (truth, score, external_miss, arrived)
        self._scored = {}       # origins with a score, as an ordered set

    def add_centers(self, t, centers):
        """Adds the centers {location: value} of origin t."""
        for location, center in centers.items():
            self._centers[location, t] = float(center)

    def add_truth(self, fb, arrived):
        """Adds one Feedback row, which arrived at step `arrived`."""
        external_miss = None if fb.external_miss is None else tuple(fb.external_miss)
        self._feedback[fb.location, fb.origin] = (float(fb.truth), float(fb.score), external_miss, arrived)
        self._scored[fb.origin] = None

    def num_scored_origins(self):
        """Number of distinct origins with a score."""
        return len(self._scored)

    def scored_origins(self):
        """The origins with a score, ascending."""
        return sorted(self._scored)

    def has_feedback(self, r, s):
        """Whether the feedback of location r at origin s has arrived."""
        return (r, s) in self._feedback

    def center(self, r, s):
        return self._centers[r, s]

    def truth(self, r, s):
        return self._feedback[r, s][0]

    def score(self, r, s):
        return self._feedback[r, s][1]

    def external_miss(self, r, s):
        """Per alpha: the delivered interval's miss (bool), or None. None when not given at all."""
        return self._feedback[r, s][2]

    def arrived(self, r, s):
        """Step at which the feedback of location r at origin s arrived."""
        return self._feedback[r, s][3]

    def normalized_score(self, r, s):
        """score / std of location r, as np.float32. The only place a score is normalized."""
        return np.float32(self.score(r, s) / self.scaler.stds[r])
