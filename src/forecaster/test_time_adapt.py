import torch


def monotonicity_measure_func1(q_hats):
    """Fraction of rows whose q-hats increase anywhere along alpha. shape of q_hats: (batch size x num alphas)"""
    q_diff = q_hats[:, 1:] - q_hats[:, :-1]
    gt_zero_counts = torch.sum(q_diff > 0, dim=1)
    gt_zero = torch.where(gt_zero_counts > 0, 1, 0)
    return torch.sum(gt_zero) / len(gt_zero)
