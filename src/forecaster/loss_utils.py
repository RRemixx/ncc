import torch
import torch.nn as nn
import numpy as np


class LossTracker:
    def __init__(self, loss_types=['qloss', 'closs', 'eloss', 'mloss']):
        self.loss_types = loss_types
        self.additional_loss_type = 'total_loss'
        self.train_loss = {}
        self.val_loss = {}
        self.tmp_train_loss = {}
        self.tmp_val_loss = {}
        for loss_type in loss_types:
            self.train_loss[loss_type] = []
            self.val_loss[loss_type] = []
        self.train_loss[self.additional_loss_type] = []
        self.val_loss[self.additional_loss_type] = []

    def add_train_loss(self, losses_in_current_epoch):
        self._add_loss(self.tmp_train_loss, losses_in_current_epoch)

    def add_val_loss(self, losses_in_current_epoch):
        self._add_loss(self.tmp_val_loss, losses_in_current_epoch)

    def _add_loss(self, tmp_loss, losses_in_current_epoch):
        qloss, closs, eloss, mloss = [loss.detach().cpu().numpy() for loss in losses_in_current_epoch]
        closs = np.nan if closs > 1e7 else closs
        tmp_loss['qloss'].append(qloss)
        tmp_loss['closs'].append(closs)
        tmp_loss['eloss'].append(eloss)
        tmp_loss['mloss'].append(mloss)

    def epoch_start(self):
        for loss_type in self.loss_types:
            self.tmp_train_loss[loss_type] = []
            self.tmp_val_loss[loss_type] = []

    def epoch_end(self, verbose=False):
        total_train_loss = 0
        total_val_loss = 0
        for loss_type in self.loss_types:
            tmp_train_loss = np.nanmean(self.tmp_train_loss[loss_type])
            tmp_val_loss = np.nanmean(self.tmp_val_loss[loss_type])
            self.train_loss[loss_type].append(tmp_train_loss)
            self.val_loss[loss_type].append(tmp_val_loss)
            total_train_loss += tmp_train_loss
            total_val_loss += tmp_val_loss
        self.train_loss[self.additional_loss_type].append(total_train_loss)
        self.val_loss[self.additional_loss_type].append(total_val_loss)
        if verbose:
            print(f'Training total loss: {total_train_loss}, cov loss: {np.mean(self.tmp_train_loss["closs"])}, quantile loss: {np.mean(self.tmp_train_loss["qloss"])}')
        return total_val_loss

    def get_losses(self):
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }


# losses
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


class LossCalculator:
    def __init__(self, training_params, alphas):
        device = torch.device(training_params['device'])
        # loss functions
        self.q_loss_fn = QuantileLoss(alphas)
        self.c_loss_fn = CoverageLoss(training_params['K'], alphas, device)
        self.e_loss_fn = EfficiencyLoss(training_params['K'])
        self.m_loss_fn = MonotonicityLoss()
        # loss factors
        self.q_loss_factor = training_params['qloss_factor']
        self.c_loss_factor = training_params['closs_factor']
        self.e_loss_factor = training_params['eloss_factor']
        self.m_loss_factor = training_params['mloss_factor']

    def cal_loss(self, cur_q_hats, cur_Ets, input_alphas, score):
        qloss = self.q_loss_factor * self.q_loss_fn(cur_q_hats, score[:, 0])
        closs = self.c_loss_factor * self.c_loss_fn(cur_Ets, cur_q_hats, score)
        eloss = self.e_loss_factor * self.e_loss_fn(cur_q_hats, score)
        mloss = self.m_loss_factor * self.m_loss_fn(cur_q_hats, input_alphas)
        loss = qloss + closs + eloss + mloss
        if torch.sum(torch.isnan(loss)) > 0:
            print(f'Loss is nan')
            print(f'qloss: {qloss}, closs: {closs}, eloss: {eloss}, mloss: {mloss}')
        return loss, qloss, closs, eloss, mloss
