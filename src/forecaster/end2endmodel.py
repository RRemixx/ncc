import torch
import torch.nn as nn
from forecaster.basicmodels import FFEncoder, ErrEncoder, RNNSeqEncoder, MultiHeadAttention


class QuantilePredictor(nn.Module):
    def __init__(self, fc_hidden_dim, hidden_dim, output_dim):
        super(QuantilePredictor, self).__init__()
        self.fc1 = nn.Linear(hidden_dim,fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.fc3 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.fc4 = nn.Linear(fc_hidden_dim, output_dim)
        self.actv = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, emb):
        """

        Args:
            emb (torch.Tensor): batch size x hidden size

        Returns:
            torch.Tensor: batch size x alphas. Non-negative and non-increasing in alpha.
        """
        x = self.actv(self.fc1(emb))
        x = self.actv(self.fc2(x))
        x = self.actv(self.fc3(x))
        x = self.relu(self.fc4(x))
        # cumulative sum from the largest alpha down, so smaller alphas get wider intervals
        x = torch.flip(x, dims=[1])
        x = torch.cumsum(x, dim=1)
        x = torch.flip(x, dims=[1])
        return x


class NCCModel(nn.Module):
    """Predicts one q-hat (half interval width) per alpha from the recent calibration history."""

    def __init__(self, params):
        super(NCCModel, self).__init__()
        self.alphas = params['alphas']
        self.params = params
        num_alphas = len(params['alphas'])
        hidden_dim = params['hidden_dim']

        self.score_encoder = RNNSeqEncoder(
            input_dim=3,
            hidden_dim=params['score_encoder_hidden_dim'],
            num_layers=params['score_encoder_num_layers'],
            output_dim=hidden_dim,
        )
        self.qhat_encoder = RNNSeqEncoder(
            input_dim=num_alphas,
            hidden_dim=params['qhat_encoder_hidden_dim'],
            num_layers=params['qhat_encoder_num_layers'],
            output_dim=hidden_dim,
        )
        self.error_encoder = ErrEncoder(
            input_dim1=num_alphas,
            input_dim2=params['window_size'],
            hidden_dim=params['error_encoder_hidden_dim'],
            output_dim=hidden_dim,
        )
        self.alpha_encoder = FFEncoder(
            input_dim=num_alphas,
            hidden_dim=params['alpha_encoder_hidden_dim'],
            output_dim=hidden_dim,
        )
        self.attn = MultiHeadAttention(
            input_dim=hidden_dim,
            num_heads=8,
        )
        self.qmodel = QuantilePredictor(
            fc_hidden_dim=params['q_fc_hidden_dim'],
            hidden_dim=hidden_dim,
            output_dim=num_alphas,
        )
        self.tta_model = FFEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_alphas,
        )

    def get_embedding(self, err_seq, score_seq, qhat_seq, input_alphas):
        score_encoding = self.score_encoder(score_seq)
        qhat_encoding = self.qhat_encoder(qhat_seq)
        error_encoding = self.error_encoder(err_seq)
        alpha_encoding = self.alpha_encoder(input_alphas)
        # fuse the four encodings with self-attention: (batch size x 4 x hidden size) -> (batch size x hidden size)
        emb = torch.stack([score_encoding, qhat_encoding, error_encoding, alpha_encoding], dim=0)
        emb = emb.permute(1, 0, 2)
        return self.attn(emb)

    def conformalize(self, current_q, err_seq, input_alphas, learning_rate, q_adj):
        """ACI-style correction: shift q-hats by the running gap between observed and target error rates.

        Args:
            current_q (torch.Tensor): model output (batch size x num alphas)
            err_seq (torch.Tensor): previous errors in a window (batch size x window size x num alphas)
            input_alphas (torch.Tensor): (batch size x num alphas)
            learning_rate (float): step size of the correction
            q_adj (torch.Tensor): accumulated correction (batch size x num alphas)
        """
        avg_err = torch.mean(err_seq, dim=1)
        new_q_adj = q_adj + learning_rate * (avg_err - input_alphas)
        adjusted_q = current_q + new_q_adj
        return adjusted_q, new_q_adj

    def forward(self, err_seq, score_seq, qhat_seq, input_alphas, learning_rate=0, q_adj=0, t=0):
        """

        Args:
            err_seq (torch.Tensor): batch size x window size x num alphas
            score_seq (torch.Tensor): batch size x window size x 3 (y_hat, y, score)
            qhat_seq (torch.Tensor): batch size x window size x num alphas
            input_alphas (torch.Tensor): batch size x num alphas
            learning_rate (float): step size of the conformal correction
            q_adj (torch.Tensor): batch size x num alphas
            t (int): current time step. The conformal correction starts after two windows.

        Returns:
            (q-hats, updated q_adj), both batch size x num alphas
        """
        emb = self.get_embedding(err_seq, score_seq, qhat_seq, input_alphas)
        qs = self.qmodel(emb)
        if t > self.params['window_size'] * 2:
            qs, q_adj = self.conformalize(qs, err_seq, input_alphas, learning_rate=learning_rate, q_adj=q_adj)
        return qs, q_adj

    def tta_forward(self, err_seq, score_seq, qhat_seq, input_alphas, learning_rate=0, q_adj=0, t=0):
        """Same as forward, plus the test-time adjustment h predicted by tta_model."""
        emb = self.get_embedding(err_seq, score_seq, qhat_seq, input_alphas)
        qs = self.qmodel(emb)
        if t > self.params['window_size'] * 2:
            qs, q_adj = self.conformalize(qs, err_seq, input_alphas, learning_rate=learning_rate, q_adj=q_adj)
        hs = self.tta_model(emb)
        return qs, q_adj, hs
