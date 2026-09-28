"""Sequential calibration: the forecasting loop with retraining, ACI and test-time adaptation (TTA)."""
import logging
from contextlib import contextmanager

import torch

from ncc.history import CalibrationHistory
from ncc.losses import MonotonicityLoss, monotonicity_fraction
from ncc.train import fit

logger = logging.getLogger(__name__)

# TTA runs when more than this fraction of the batch has q-hats that increase with alpha
TTA_MONOTONICITY_THRESHOLD = 0.05
TTA_MAX_STEPS = 200
TTA_CHECK_EVERY = 10    # TTA checks every this many steps whether it can stop


def calibrate(model, data, scaler, cfg, regions, horizon, first_test, max_steps):
    """Runs the trained model over every step and returns the test forecasts in the original units.

    The history is filled from step 0, so it is complete at the first test step. From first_test on,
    the model is retrained every cfg.retrain.period steps on [0, t - val_steps); the forecast of step t
    is made before that retrain. The history is never rebuilt after a retrain.
    Stops after step max_steps.
    Returns results[region][horizon] = [(q_hats, y_hat, y), ...], one entry per test step.
    """
    device = torch.device(cfg.device)
    num_steps = len(data[regions[0]])
    model = model.to(device)
    model.eval()
    results = {region: {horizon: []} for region in regions}
    history = CalibrationHistory(regions, cfg.alphas, cfg.window_size + num_steps + horizon - 1,
                                 cfg.init_err_with_alpha)
    for t in range(min(num_steps, max_steps + 1)):
        q_hats, batch = _predict(model, history, data, regions, t, cfg, horizon, device)
        if t < first_test:
            continue
        if (t - first_test + 1) % cfg.retrain.period == 0:
            _retrain(model, data, cfg, regions, horizon, t)
            model.eval()
        _save_forecast(results, horizon, scaler, q_hats, batch)
    return results


def _retrain(model, data, cfg, regions, horizon, t):
    """Retrains on [0, t - val_steps), with the gradient-free walk up to t, on the retrain schedule."""
    logger.info('retrain started at step %d', t)
    schedule = cfg.retrain.epochs if cfg.train.stages is None else cfg.retrain.stage_epochs
    fit(model, data, cfg, regions, horizon, train_end=t - cfg.val_steps, walk_end=t, schedule=schedule)
    logger.info('retrain finished')


def _save_forecast(results, horizon, scaler, q_hats, batch):
    """Appends the forecast of one step, in the original units, to results."""
    q_hats = q_hats.detach().cpu().numpy()
    y_hats = batch.y_hats.detach().cpu().numpy()
    ys = batch.ys.detach().cpu().numpy()
    for r, region in enumerate(batch.regions):
        q_hat = scaler.inverse(region, q_hats[r, :], is_score=True)
        y_hat = scaler.inverse(region, y_hats[r])
        y = scaler.inverse(region, ys[r])
        results[region][horizon].append((q_hat, y_hat, y))


def _predict(model, history, data, regions, t, cfg, horizon, device):
    """q-hats of step t (with TTA and ACI if enabled), written into the history. Returns (q_hats, batch)."""
    batch = history.batch(data, regions, t, cfg.window_size, device)
    if cfg.tta.enabled:
        q_hats, q_adj = _tta_predict(model, batch, t, cfg)
    else:
        # TODO(paper): ACI only runs inside TTA
        q_hats = model(batch.err_seq, batch.score_seq, batch.qhat_seq, batch.input_alphas)
        q_adj = batch.q_adj
    history.update(t + cfg.window_size + horizon - 1, batch, q_hats, q_adj)
    return q_hats, batch


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
