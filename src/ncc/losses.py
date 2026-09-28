"""NCC losses: quantile, coverage, efficiency and monotonicity, their weighted sum, and the monotonicity check of TTA."""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantileLoss(nn.Module):
    def __init__(self, alphas):
        super().__init__()
        self.quantiles = []
        for alpha in alphas:
            self.quantiles.append(1-alpha)

    def forward(self, preds, target):
        # assume preds in the shape of (batch size, quantiles size), target in the shape of (batch size)
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        assert preds.size(1) == len(self.quantiles)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target[:] - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class CoverageLoss(nn.Module):
    def __init__(self, K, alphas, device):
        """
        K is used in sigmoid function.
        alphas: (A)
        """
        super().__init__()
        self.K = K
        self.alphas = torch.Tensor(alphas).to(device)

    def forward(self, cur_Ets, q_hat, q):
        """
        Denote the number of alphas by A, the number of batches by B, the window size is W.
        cur_Ets: errors in a fixed window. (B x W x A)
        q_hat: prediction intervals. (B x A)
        q: uncertainty scores, |y - y_hat|. (B x 1)
        """
        e_prev = torch.mean(cur_Ets, dim=1)
        e_hat = torch.sigmoid((q - q_hat) / self.K)
        # clamp so that e_hat is within a valid range
        e_hat = torch.clamp(e_hat, min=0.01, max=0.99)
        tmp = self.alphas[None, :] - e_prev
        e = torch.where(tmp > 0, 1, 0)
        closs = -1 * torch.nanmean(e*torch.log(e_hat) + (1-e)*torch.log(1-e_hat))
        return closs


class EfficiencyLoss(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, q_hat, q):
        """
        Calculate efficiency loss.
        """
        # current error estimated by sigmoid function
        cur_error = torch.sigmoid((q - q_hat) / self.K)

        # efficiency loss
        eloss = torch.square(q_hat - q) * (1 - cur_error)
        return torch.mean(eloss)


class MonotonicityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_hat, input_alphas):
        """
        Calculate monotonicity loss: penalize q-hats that increase with alpha.
        Denote the number of alphas by A, the number of batches by B.
        q_hat: prediction intervals. (B x A)
        input_alphas: alphas. (B x A)
        """
        q_diff = torch.diff(q_hat, dim=1)
        a_diff = torch.diff(input_alphas, dim=1)
        dev = q_diff / a_diff
        mloss = nn.functional.relu(dev)
        return torch.mean(mloss)


class NCCLoss:
    """Weighted sum of the four losses. Each part is already multiplied by its weight."""

    def __init__(self, weights, K, alphas, device):
        """weights: LossWeights; K: sigmoid temperature; alphas: target miscoverage rates; device: torch device."""
        self.q_loss_fn = QuantileLoss(alphas)
        self.c_loss_fn = CoverageLoss(K, alphas, device)
        self.e_loss_fn = EfficiencyLoss(K)
        self.m_loss_fn = MonotonicityLoss()
        self.weights = weights

    def __call__(self, q_hat, err_seq, alphas, score):
        """Returns (total, (quantile, coverage, efficiency, monotonicity)).

        q_hat: (B x A); err_seq: (B x W x A); alphas: (B x A); score: |y - y_hat| (B x 1).
        """
        w = self.weights
        qloss = w.quantile * self.q_loss_fn(q_hat, score[:, 0])
        closs = w.coverage * self.c_loss_fn(err_seq, q_hat, score)
        eloss = w.efficiency * self.e_loss_fn(q_hat, score)
        mloss = w.monotonicity * self.m_loss_fn(q_hat, alphas)
        total = qloss + closs + eloss + mloss
        if torch.sum(torch.isnan(total)) > 0:
            logger.warning('Loss is nan (qloss: %s, closs: %s, eloss: %s, mloss: %s)', qloss, closs, eloss, mloss)
        return total, (qloss, closs, eloss, mloss)


def monotonicity_fraction(q_hats):
    """Fraction of rows whose q-hats increase anywhere along alpha. shape of q_hats: (batch size x num alphas)"""
    q_diff = q_hats[:, 1:] - q_hats[:, :-1]
    gt_zero_counts = torch.sum(q_diff > 0, dim=1)
    gt_zero = torch.where(gt_zero_counts > 0, 1, 0)
    return torch.sum(gt_zero) / len(gt_zero)
