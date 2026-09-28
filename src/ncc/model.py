"""The NCC model: encoders of the calibration history, self-attention fusion and the quantile head."""
import torch
import torch.nn as nn

from ncc.config import ModelConfig
from ncc.layers import FFEncoder, ErrEncoder, RNNSeqEncoder, MultiHeadAttention


class QuantileHead(nn.Module):
    def __init__(self, fc_hidden_dim, hidden_dim, output_dim):
        super(QuantileHead, self).__init__()
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
    """Predicts one normalized q-hat (half interval width) per alpha from the recent calibration history.

    The attributes are created in a fixed order (score, qhat, error and alpha encoders, fusion,
    quantile head, TTA head): changing it changes the random initialization.
    """

    def __init__(self, cfg: ModelConfig, num_alphas: int, window_size: int):
        super(NCCModel, self).__init__()
        hidden_dim = cfg.hidden_dim
        self.score_encoder = RNNSeqEncoder(
            input_dim=3,
            hidden_dim=cfg.score_encoder_hidden_dim,
            num_layers=cfg.score_encoder_num_layers,
            output_dim=hidden_dim,
        )
        self.qhat_encoder = RNNSeqEncoder(
            input_dim=num_alphas,
            hidden_dim=cfg.qhat_encoder_hidden_dim,
            num_layers=cfg.qhat_encoder_num_layers,
            output_dim=hidden_dim,
        )
        self.error_encoder = ErrEncoder(
            input_dim1=num_alphas,
            input_dim2=window_size,
            hidden_dim=cfg.error_encoder_hidden_dim,
            output_dim=hidden_dim,
        )
        self.alpha_encoder = FFEncoder(
            input_dim=num_alphas,
            hidden_dim=cfg.alpha_encoder_hidden_dim,
            output_dim=hidden_dim,
        )
        self.fusion = MultiHeadAttention(
            input_dim=hidden_dim,
            num_heads=8,
        )
        self.quantile_head = QuantileHead(
            fc_hidden_dim=cfg.q_fc_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_alphas,
        )
        self.tta_head = FFEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_alphas,
        )

    def embed(self, err_seq, score_seq, qhat_seq, alphas):
        """Fused embedding (batch size x hidden size) of the four encodings.

        Args:
            err_seq (torch.Tensor): batch size x window size x num alphas
            score_seq (torch.Tensor): batch size x window size x 3 (y_hat, y, score)
            qhat_seq (torch.Tensor): batch size x window size x num alphas
            alphas (torch.Tensor): batch size x num alphas
        """
        score_encoding = self.score_encoder(score_seq)
        qhat_encoding = self.qhat_encoder(qhat_seq)
        error_encoding = self.error_encoder(err_seq)
        alpha_encoding = self.alpha_encoder(alphas)
        # fuse the four encodings with self-attention: (batch size x 4 x hidden size) -> (batch size x hidden size).
        # Keep stack + permute: another memory layout changes the matmul summation order, and training amplifies that.
        emb = torch.stack([score_encoding, qhat_encoding, error_encoding, alpha_encoding], dim=0)
        emb = emb.permute(1, 0, 2)
        return self.fusion(emb)

    def forward(self, err_seq, score_seq, qhat_seq, alphas):
        """Normalized q-hats (batch size x num alphas); see embed for the inputs."""
        return self.quantile_head(self.embed(err_seq, score_seq, qhat_seq, alphas))
