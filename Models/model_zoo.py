from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    """One-sublayer TCN block: causal conv -> LayerNorm -> SiLU -> dropout -> residual."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = F.silu(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        return x + y


class CEMATCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 6, kernel_size: int = 5, dropout: float = 0.04) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden_dim, kernel_size=kernel_size, dilation=2**idx, dropout=dropout) for idx in range(layers)]
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input(x.transpose(1, 2))
        for block in self.blocks:
            y = block(y)
        endpoint = y[:, :, -1]
        return self.head(endpoint).squeeze(-1)


class RecurrentRegressor(nn.Module):
    def __init__(self, cell: str, input_dim: int, hidden_dim: int = 128, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, layers: int = 2, dropout: float = 0.04) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(self.input(x))
        return self.head(y[:, -1]).squeeze(-1)


class EndpointMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.04) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, -1]).squeeze(-1)


def make_model(model: str, input_dim: int, hidden_dim: int = 128, layers: int | None = None) -> nn.Module:
    model = model.lower()
    if model in {"cema_tcn", "tcn"}:
        layers = 6 if layers is None else layers
        return CEMATCN(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers)
    if model == "lstm":
        layers = 1 if layers is None else layers
        return RecurrentRegressor("lstm", input_dim=input_dim, hidden_dim=hidden_dim, layers=layers)
    if model == "gru":
        layers = 1 if layers is None else layers
        return RecurrentRegressor("gru", input_dim=input_dim, hidden_dim=hidden_dim, layers=layers)
    if model == "transformer":
        layers = 2 if layers is None else layers
        return TransformerRegressor(input_dim=input_dim, d_model=hidden_dim, nhead=8, layers=layers)
    if model == "mlp":
        return EndpointMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown model: {model}")
