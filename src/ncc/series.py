"""Per-location series of (point forecast, truth) for one horizon, normalized with a per-location scaler."""
import numpy as np


class Scaler:
    """Standardizes y and y_hat per location. Scores (|y - y_hat|) are only rescaled."""

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, location, truths):
        """Fits the mean and std of one location on its truths (an array)."""
        self.stds[location] = np.std(truths)
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


def build_series(base_pred, horizon, train_steps):
    """Normalized series of one horizon, and the scaler fitted on the truths of steps [0, train_steps).

    base_pred: one (predictions, addition_infos) pair per time step (see README, base prediction format).
    horizon: steps ahead, counted from 1.
    Returns (data, scaler) with data[location] = [(y_hat_n, y_n), ...], one pair per time step.
    """
    locations = list(base_pred[0][0].keys())
    series = {location: [] for location in locations}
    for predictions, addition_infos in base_pred:
        for location in locations:
            y = addition_infos[location][0]
            # horizon is 1-based, predictions are indexed from 0
            series[location].append((predictions[location][horizon - 1], y[horizon - 1]))
    scaler = Scaler()
    data = {}
    for location in locations:
        steps = series[location]
        scaler.fit(location, np.array([y for _, y in steps[:train_steps]]))
        data[location] = [(scaler.transform(location, y_hat), scaler.transform(location, y)) for y_hat, y in steps]
    return data, scaler
