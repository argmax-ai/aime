import torch

from aime.dist import Normal, TanhNormal

from .base import MIN_STD, MLP


class TanhGaussianPolicy(torch.nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_size=32, hidden_layers=2, min_std=None
    ) -> None:
        super().__init__()
        self.min_std = min_std if min_std is not None else MIN_STD
        self.mean_net = MLP(state_dim, action_dim, hidden_size, hidden_layers)
        self.std_net = MLP(
            state_dim,
            action_dim,
            hidden_size,
            hidden_layers,
            output_activation="softplus",
        )

    def forward(self, state):
        mean = self.mean_net(state)
        std = self.std_net(state) + self.min_std
        return TanhNormal(mean, std)


class GaussianPolicy(torch.nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_size=32, hidden_layers=2, min_std=None
    ) -> None:
        super().__init__()
        self.min_std = min_std if min_std is not None else MIN_STD
        self.mean_net = MLP(state_dim, action_dim, hidden_size, hidden_layers)
        self.std_net = MLP(
            state_dim,
            action_dim,
            hidden_size,
            hidden_layers,
            output_activation="softplus",
        )

    def forward(self, state):
        mean = self.mean_net(state)
        std = self.std_net(state) + self.min_std
        return Normal(mean, std)
