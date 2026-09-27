# TODO: now using training loss as val loss

from typing import NamedTuple

import torch
import numpy as np

from forecaster.utils import EarlyStopping
from forecaster.loss_utils import LossCalculator, LossTracker, MonotonicityLoss
from forecaster.end2endmodel import NCCModel
from forecaster.test_time_adapt import monotonicity_measure_func1

# TTA runs when more than this fraction of the batch has q-hats that increase with alpha
MONOTONICITY_THRESHOLD = 0.05
TTA_MAX_STEPS = 200


class End2endCPDataset:
    def __init__(self, base_pred, week_ahead_list):
        """base_pred: one (predictions, addition_infos) pair per time step, as saved by the base forecaster."""
        self.base_pred = base_pred
        self.week_ahead_list = week_ahead_list
        self.regions = list(base_pred[0][0].keys())

    def e2ecp_seq_data(self, training_params):
        """Returns seq_data[region][week_ahead] = [(y_hat, y), ...] normalized per region and week ahead, and the scalar."""
        seq_data = {region: {week_ahead: [] for week_ahead in self.week_ahead_list} for region in self.regions}
        for predictions, addition_infos in self.base_pred:
            for region in self.regions:
                y = addition_infos[region][0]
                for week_ahead in self.week_ahead_list:
                    # week ahead is 1-based, predictions are indexed from 0
                    seq_data[region][week_ahead].append((predictions[region][week_ahead - 1], y[week_ahead - 1]))
        # normalize with the statistics of the training steps
        scalar = E2ecpScalar(self.regions)
        data_size = len(seq_data[self.regions[0]][self.week_ahead_list[0]])
        training_steps = data_size - training_params['val_weeks'] - training_params['test_weeks']
        for region in self.regions:
            for week_ahead in self.week_ahead_list:
                steps = seq_data[region][week_ahead]
                scalar.fit(np.array([y for _, y in steps[:training_steps]]), region, week_ahead)
                seq_data[region][week_ahead] = [
                    (scalar.transform(y_hat, region, week_ahead), scalar.transform(y, region, week_ahead))
                    for y_hat, y in steps
                ]
        return seq_data, scalar

    def get_regions(self):
        return self.regions


class E2ecpScalar:
    """Standardizes y and y_hat per region and week ahead. Scores (|y - y_hat|) are only rescaled."""

    def __init__(self, regions):
        self.stds = {region: {} for region in regions}
        self.means = {region: {} for region in regions}

    def fit(self, data, region, week_ahead):
        self.stds[region][week_ahead] = np.std(data)
        self.means[region][week_ahead] = np.mean(data)

    def transform(self, data, region, week_ahead):
        data = data - self.means[region][week_ahead]
        data = data / self.stds[region][week_ahead]
        return data

    def inv_transform(self, data, region, week_ahead, is_score=False):
        data = data * self.stds[region][week_ahead]
        if not is_score:
            data = data + self.means[region][week_ahead]
        return data


def init_model(model_params, alphas, cov_window_size):
    return NCCModel({**model_params, 'alphas': alphas, 'window_size': cov_window_size})


class Batch(NamedTuple):
    """Model inputs for one time step. B = regions, W = cov window size, A = alphas."""
    regions: list               # region of each row
    y_hats: torch.Tensor        # (B x 1)
    ys: torch.Tensor            # (B x 1)
    scores: torch.Tensor        # (B x 1), |y - y_hat|
    err_seq: torch.Tensor       # (B x W x A)
    score_seq: torch.Tensor     # (B x W x 3): y_hat, y, score
    qhat_seq: torch.Tensor      # (B x W x A)
    input_alphas: torch.Tensor  # (B x A)
    q_adj: torch.Tensor         # (B x A)


class RollingSequences:
    """Per-region calibration history that the model reads through a sliding window.

    The outcome of the forecast made at step t is stored at position t + W + week_ahead - 1,
    so the window [t, t + W) only contains outcomes that are already observed.
    """

    def __init__(self, regions, alphas, length, init_err_seq_w_alpha):
        self.alphas = alphas
        self.errors = {}        # (A x length): 1 if y fell outside the q-hat interval
        self.scores = {}        # (length)
        self.preds_truths = {}  # (length x 2): y_hat, y
        self.q_adjs = {}        # (A): accumulated conformal correction
        for region in regions:
            errors = torch.ones((len(alphas), length))
            if init_err_seq_w_alpha:
                errors = errors * torch.tensor(alphas)[:, None]
            self.errors[region] = errors
            self.scores[region] = torch.zeros(length)
            self.preds_truths[region] = torch.zeros((length, 2))
            self.q_adjs[region] = torch.zeros(len(alphas))

    def batch(self, data, regions, t, cov_window_size, week_ahead, device):
        region_idxs = np.arange(len(regions))
        np.random.shuffle(region_idxs)
        batch_regions = [regions[i] for i in region_idxs]
        window = slice(t, t + cov_window_size)
        y_hats = torch.Tensor([[data[region][week_ahead][t][0]] for region in batch_regions]).to(device)
        ys = torch.Tensor([[data[region][week_ahead][t][1]] for region in batch_regions]).to(device)
        err_seq = torch.stack([self.errors[region][:, window].T for region in batch_regions]).to(device)
        score_seq = torch.stack([
            torch.cat([self.preds_truths[region][window], self.scores[region][window, None]], dim=1)
            for region in batch_regions
        ]).to(device)
        # the q-hat history was never written in the paper runs, so the q-hat encoder always sees zeros
        qhat_seq = torch.zeros((len(regions), cov_window_size, len(self.alphas))).to(device)
        input_alphas = torch.tensor(self.alphas)[None, :].repeat(len(regions), 1).to(device)
        q_adj = torch.stack([self.q_adjs[region] for region in batch_regions]).to(device)
        scores = torch.abs(y_hats - ys)
        return Batch(batch_regions, y_hats, ys, scores, err_seq, score_seq, qhat_seq, input_alphas, q_adj)

    def update(self, idx, batch, q_hats, q_adj):
        for r, region in enumerate(batch.regions):
            self.preds_truths[region][idx, 0] = batch.y_hats[r, 0]
            self.preds_truths[region][idx, 1] = batch.ys[r, 0]
            self.q_adjs[region] = q_adj[r]
            self.errors[region][:, idx] = (q_hats[r] < batch.scores[r]).cpu().float()
            self.scores[region][idx] = batch.scores[r, 0]


class End2endCPTrainerForecaster:
    def __init__(self, model: NCCModel, regions, week_ahead_list, alphas, scalar: E2ecpScalar, cf_lr, tta_reg_factor, tta_lr):
        self.model = model
        self.regions = regions
        self.week_ahead_list = week_ahead_list
        self.alphas = alphas
        self.scalar = scalar
        self.cf_lr = cf_lr

        # tta
        self.tta_reg_factor = tta_reg_factor
        self.tta_lr = tta_lr
        self.mono_loss = MonotonicityLoss()

    def new_sequences(self, training_params, data_size, week_ahead):
        length = training_params['cov_window_size'] + data_size + week_ahead - 1
        return RollingSequences(self.regions, self.alphas, length, training_params['init_err_seq_w_alpha'])

    def forward_pass(self, data, t, cov_window_size, week_ahead, device, seqs):
        batch = seqs.batch(data, self.regions, t, cov_window_size, week_ahead, device)
        # t is not passed, so the conformal correction stays off here; it only runs inside TTA
        q_hats, q_adj = self.model(batch.err_seq, batch.score_seq, batch.qhat_seq, batch.input_alphas, q_adj=batch.q_adj)
        seqs.update(t + cov_window_size + week_ahead - 1, batch, q_hats, q_adj)
        return q_hats, batch

    def tta_forward_pass(self, data, t, cov_window_size, week_ahead, device, seqs):
        """Forward pass that fixes non-monotone q-hats by fitting the TTA head (all other weights frozen)."""
        batch = seqs.batch(data, self.regions, t, cov_window_size, week_ahead, device)
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.tta_lr)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.tta_model.parameters():
            param.requires_grad = True
        inputs = (batch.err_seq, batch.score_seq, batch.qhat_seq, batch.input_alphas)
        q_hats, q_adj, hs = model.tta_forward(*inputs, learning_rate=self.cf_lr, q_adj=batch.q_adj, t=t)
        if monotonicity_measure_func1(q_hats) > MONOTONICITY_THRESHOLD:
            for step in range(1, TTA_MAX_STEPS + 1):
                q_hats, q_adj, hs = model.tta_forward(*inputs, learning_rate=self.cf_lr, q_adj=batch.q_adj, t=t)
                q_hats = q_hats + hs
                if step % 10 == 0 and monotonicity_measure_func1(q_hats) < MONOTONICITY_THRESHOLD:
                    break
                optimizer.zero_grad()
                loss_m = self.mono_loss(q_hats, batch.input_alphas)
                loss_reg = hs.pow(2).mean().sqrt()
                tta_loss = loss_m + loss_reg * self.tta_reg_factor
                tta_loss.backward()
                optimizer.step()
        seqs.update(t + cov_window_size + week_ahead - 1, batch, q_hats, q_adj)
        # unfreeze parameters of model
        for param in model.parameters():
            param.requires_grad = True
        return q_hats, batch

    def train(self, training_params, data):
        device = torch.device(training_params['device'])
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['lr'])
        loss_calculator = LossCalculator(training_params, self.alphas)
        loss_tracker = LossTracker()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=training_params['gamma'])
        early_stopping = EarlyStopping(patience=50)

        cov_window_size = training_params['cov_window_size']
        data_size = len(data[self.regions[0]][self.week_ahead_list[0]])
        num_train_weeks = data_size - training_params['val_weeks'] - training_params['test_weeks']
        for epoch in range(training_params['epochs']):
            loss_tracker.epoch_start()
            for week_ahead in self.week_ahead_list:
                seqs = self.new_sequences(training_params, data_size, week_ahead)
                # train
                model.train()
                for t in range(num_train_weeks):
                    q_hats, batch = self.forward_pass(data, t, cov_window_size, week_ahead, device, seqs)
                    loss, *losses = loss_calculator.cal_loss(q_hats, batch.err_seq, batch.input_alphas, batch.scores)
                    loss_tracker.add_train_loss(losses)
                    loss_tracker.add_val_loss(losses)
                    # backward pass
                    optimizer.zero_grad()
                    try:
                        loss.backward()
                        # gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                        optimizer.step()
                    except RuntimeError:
                        print('Infinite loss, skipped!')
                # the validation weeks only advance the rolling sequences (see the TODO at the top)
                model.eval()
                for t in range(num_train_weeks, num_train_weeks + training_params['val_weeks']):
                    self.forward_pass(data, t, cov_window_size, week_ahead, device, seqs)
            val_loss = loss_tracker.epoch_end(verbose=True)
            scheduler.step()
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        if training_params['epochs'] > 0:
            model.load_state_dict(early_stopping.model_state_dict)
        return loss_tracker.get_losses()

    def stage_train(self, training_params, data):
        """Train once, or in three stages with the per-stage loss factors in training_params['loss_factors']."""
        if not training_params['three_stage_training']:
            return self.train(training_params, data)
        for epochs, (qloss_factor, closs_factor, eloss_factor) in zip(training_params['epochs_in_stage'], training_params['loss_factors']):
            stage_params = {
                **training_params,
                'epochs': int(epochs),
                'qloss_factor': qloss_factor,
                'closs_factor': closs_factor,
                'eloss_factor': eloss_factor,
            }
            losses = self.train(stage_params, data)
        return losses

    def retrain(self, training_params, data, steps_after_last_val):
        """Retrain on every step observed so far, with the shorter retrain schedule."""
        print('retrain started')
        retrain_params = {
            **training_params,
            'test_weeks': training_params['test_weeks'] - steps_after_last_val,
            'epochs': training_params['retrain_epoch'],
            'epochs_in_stage': training_params['retrain_epochs'],
        }
        self.stage_train(retrain_params, data)
        print('retrain finished')

    def forecast4eval(self, training_params, data):
        """
        Runs through all steps, so the rolling sequences are filled before the test weeks, and returns
        predictions[region][week_ahead] = [(q_hats, y_hat, y), ...] for the test weeks in the original scale.
        """
        device = torch.device(training_params['device'])
        cov_window_size = training_params['cov_window_size']
        data_size = len(data[self.regions[0]][self.week_ahead_list[0]])
        first_test_step = data_size - training_params['test_weeks']
        forward = self.tta_forward_pass if training_params['use_tta'] else self.forward_pass
        model = self.model.to(device)
        model.eval()

        all_predictions = {region: {week_ahead: [] for week_ahead in self.week_ahead_list} for region in self.regions}
        for week_ahead in self.week_ahead_list:
            seqs = self.new_sequences(training_params, data_size, week_ahead)
            for t in range(min(data_size, training_params['total_steps_limit'] + 1)):
                q_hats, batch = forward(data, t, cov_window_size, week_ahead, device, seqs)
                if t < first_test_step:
                    continue
                steps_after_val = t - first_test_step + 1
                if steps_after_val % training_params['retrain_period'] == 0:
                    self.retrain(training_params, data, steps_after_val - 1)
                    model.eval()
                # save the forecast made before retraining, in the original scale
                q_hats = q_hats.detach().cpu().numpy()
                y_hats = batch.y_hats.detach().cpu().numpy()
                ys = batch.ys.detach().cpu().numpy()
                for r, region in enumerate(batch.regions):
                    q_hat = self.scalar.inv_transform(q_hats[r, :], region, week_ahead, is_score=True)
                    y_hat = self.scalar.inv_transform(y_hats[r], region, week_ahead)
                    y = self.scalar.inv_transform(ys[r], region, week_ahead)
                    all_predictions[region][week_ahead].append((q_hat, y_hat, y))
        return all_predictions
