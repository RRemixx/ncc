"""Calibration history: the past misses, scores and forecasts the model reads through a sliding window."""
from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    """Model inputs for one time step. B = locations, W = window size, A = alphas."""
    regions: list               # location of each row
    y_hats: torch.Tensor        # (B x 1)
    ys: torch.Tensor            # (B x 1)
    scores: torch.Tensor        # (B x 1), |y - y_hat|
    err_seq: torch.Tensor       # (B x W x A)
    score_seq: torch.Tensor     # (B x W x 3): y_hat, y, score
    qhat_seq: torch.Tensor      # (B x W x A)
    input_alphas: torch.Tensor  # (B x A)
    q_adj: torch.Tensor         # (B x A)


class CalibrationHistory:
    """Per-location calibration history that the model reads through a sliding window.

    The outcome of the forecast made at step t is stored at slot t + W + horizon - 1,
    so the window [t, t + W) only contains outcomes that are already observed.
    Slots that were never written hold placeholders: miss = 1 (or alpha), score = 0, (y_hat, y) = 0.
    """

    def __init__(self, regions, alphas, length, init_err_with_alpha):
        self.alphas = alphas
        self.errors = {}        # (A x length): 1 if y fell outside the q-hat interval
        self.scores = {}        # (length)
        self.preds_truths = {}  # (length x 2): y_hat, y
        self.q_adjs = {}        # (A): accumulated conformal correction
        for region in regions:
            errors = torch.ones((len(alphas), length))
            if init_err_with_alpha:
                errors = errors * torch.tensor(alphas)[:, None]
            self.errors[region] = errors
            self.scores[region] = torch.zeros(length)
            self.preds_truths[region] = torch.zeros((length, 2))
            self.q_adjs[region] = torch.zeros(len(alphas))

    def batch(self, data, regions, t, window_size, device):
        """Batch of step t: all regions in a new random order (np.random), each with the window [t, t + W).

        data[region][t] = (y_hat_n, y_n).
        """
        region_idxs = np.arange(len(regions))
        np.random.shuffle(region_idxs)
        batch_regions = [regions[i] for i in region_idxs]
        window = slice(t, t + window_size)
        y_hats = torch.Tensor([[data[region][t][0]] for region in batch_regions]).to(device)
        ys = torch.Tensor([[data[region][t][1]] for region in batch_regions]).to(device)
        err_seq = torch.stack([self.errors[region][:, window].T for region in batch_regions]).to(device)
        score_seq = torch.stack([
            torch.cat([self.preds_truths[region][window], self.scores[region][window, None]], dim=1)
            for region in batch_regions
        ]).to(device)
        # TODO(paper): the q-hat history is never written, so the q-hat encoder always sees zeros
        qhat_seq = torch.zeros((len(regions), window_size, len(self.alphas))).to(device)
        input_alphas = torch.tensor(self.alphas)[None, :].repeat(len(regions), 1).to(device)
        q_adj = torch.stack([self.q_adjs[region] for region in batch_regions]).to(device)
        scores = torch.abs(y_hats - ys)
        return Batch(batch_regions, y_hats, ys, scores, err_seq, score_seq, qhat_seq, input_alphas, q_adj)

    def update(self, idx, batch, q_hats, q_adj):
        """Writes the outcome of the batch's step into slot idx; a miss is q_hat < score."""
        for r, region in enumerate(batch.regions):
            self.preds_truths[region][idx, 0] = batch.y_hats[r, 0]
            self.preds_truths[region][idx, 1] = batch.ys[r, 0]
            self.q_adjs[region] = q_adj[r]
            self.errors[region][:, idx] = (q_hats[r] < batch.scores[r]).cpu().float()
            self.scores[region][idx] = batch.scores[r, 0]
