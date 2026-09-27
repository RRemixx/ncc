import torch
import torch.nn as nn


class FFEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): batch size x input dim

        Returns:
            torch.Tensor: batch size x output dim
        """
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class ErrEncoder(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super(ErrEncoder, self).__init__()
        self.shape1 = input_dim2 * hidden_dim
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(input_dim2*hidden_dim, output_dim)
        self.actv = nn.Tanh()

    def forward(self, Ets):
        """

        Args:
            Ets (torch.Tensor): batch size x coverage window size x alphas

        Returns:
            torch.Tensor: batch size x output dim
        """
        x = self.actv(self.fc1(Ets))
        x = self.actv(self.fc2(x)).reshape(-1, self.shape1)
        x = self.actv(self.fc3(x))
        return x


class RNNSeqEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNSeqEncoder, self).__init__()
        self.output_dim = output_dim
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(num_layers, 1)
        self.actv = nn.Tanh()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): batch size x sequence length x input dim

        Returns:
            torch.Tensor: batch size x output dim
        """
        x = self.rnn(x)[-1]
        x = x.permute(1, 2, 0)
        x = self.actv(self.fc3(x))[:, :, 0]
        x = self.actv(self.fc1(x))
        x = self.fc2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): batch size x sequence length x input dim

        Returns:
            torch.Tensor: batch size x input dim (summed over the sequence)
        """
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        weighted = torch.sum(weighted, dim=1)
        return weighted


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(input_dim=input_dim)
             for _ in range(num_heads)]
        )
        self.fc = nn.Linear(num_heads, 1)

    def forward(self, x):
        embs = torch.stack([head(x) for head in self.heads], dim=-1)
        return self.fc(embs)[:, :, 0]
