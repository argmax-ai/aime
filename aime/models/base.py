import torch
import torchvision
from torch import nn
from torch.functional import F

from aime.dist import Normal

MIN_STD = 1e-6


class MLP(nn.Module):
    r"""
    Multi-layer Perceptron
    Inputs:
        in_features : int, features numbers of the input
        out_features : int, features numbers of the output
        hidden_size : int, features numbers of the hidden layers
        hidden_layers : int, numbers of the hidden layers
        norm : str, normalization method between hidden layers, default : None
        hidden_activation : str, activation function used in hidden layers, default : 'leakyrelu'
        output_activation : str, activation function used in output layer, default : 'identity'
    """  # noqa: E501

    ACTIVATION_CREATORS = {
        "relu": lambda: nn.ReLU(inplace=True),
        "elu": lambda: nn.ELU(),
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
        "identity": lambda: nn.Identity(),
        "gelu": lambda: nn.GELU(),
        "swish": lambda: nn.SiLU(),
        "softplus": lambda: nn.Softplus(),
    }

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        hidden_size: int = 32,
        hidden_layers: int = 2,
        norm: str = None,
        have_head: bool = True,
        hidden_activation: str = "elu",
        output_activation: str = "identity",
        zero_init: bool = False,
    ):
        super(MLP, self).__init__()

        if out_features is None:
            out_features = hidden_size
        self.output_dim = out_features

        hidden_activation_creator = self.ACTIVATION_CREATORS[hidden_activation]
        output_activation_creator = self.ACTIVATION_CREATORS[output_activation]

        if hidden_layers == 0:
            assert have_head, "you have to have a head when there is no hidden layers!"
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features), output_activation_creator()
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(
                    nn.Linear(in_features if i == 0 else hidden_size, hidden_size)
                )
                if norm:
                    if norm == "ln":
                        net.append(nn.LayerNorm(hidden_size))
                    elif norm == "bn":
                        net.append(nn.BatchNorm1d(hidden_size))
                    else:
                        raise NotImplementedError(f"{norm} does not supported!")
                net.append(hidden_activation_creator())
            if have_head:
                net.append(nn.Linear(hidden_size, out_features))
                if zero_init:
                    with torch.no_grad():
                        net[-1].weight.fill_(0)
                        net[-1].bias.fill_(0)
                net.append(output_activation_creator())
            self.net = nn.Sequential(*net)

    def forward(self, x):
        r"""forward method of MLP only assume the last dim of x matches `in_features`"""
        head_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = self.net(x)
        out = out.view(*head_shape, out.shape[-1])
        return out


class CNNEncoderHa(nn.Module):
    """
    The structure is introduced in Ha and Schmidhuber, World Model.
    NOTE: The structure only works for 64 x 64 image.
    """

    def __init__(self, image_size, width=32, *args, **kwargs) -> None:
        super().__init__()

        self.resize = torchvision.transforms.Resize(64)
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 4, 2),
            nn.ReLU(True),  # This relu is problematic
            nn.Conv2d(width, width * 2, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(width * 2, width * 4, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(width * 4, width * 8, 4, 2),
            nn.Flatten(start_dim=-3, end_dim=-1),
        )

        self.output_dim = 4 * width * 8

    def forward(self, image):
        """forward process an image, the return feature is 1024 dims"""
        head_dims = image.shape[:-3]
        image = image.view(-1, *image.shape[-3:])
        image = self.resize(image)
        output = self.net(image)
        return output.view(*head_dims, self.output_dim)


class IndentityEncoder(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim

    def forward(self, x):
        return x


encoder_classes = {
    "mlp": MLP,
    "identity": IndentityEncoder,
    "cnn_ha": CNNEncoderHa,
}


class MultimodalEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoders = torch.nn.ModuleDict()
        for name, dim, encoder_config in self.config:
            encoder_config = encoder_config.copy()
            encoder_type = encoder_config.pop("name")
            self.encoders[name] = encoder_classes[encoder_type](dim, **encoder_config)

        self.output_dim = sum(
            [encoder.output_dim for name, encoder in self.encoders.items()]
        )

    def forward(self, obs):
        return torch.cat(
            [model(obs[name]) for name, model in self.encoders.items()], dim=-1
        )


class MLPDeterministicDecoder(torch.nn.Module):
    r"""
    determinasticly decode the states to outputs.
    For consistent API, it output a Guassian with \sigma=1,
    so that the gradient is the same as L2 loss.
    """

    def __init__(
        self,
        state_dim,
        obs_dim,
        hidden_size,
        hidden_layers,
        norm=None,
        hidden_activation="elu",
    ) -> None:
        super().__init__()
        self.net = MLP(
            state_dim,
            obs_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation=hidden_activation,
        )

    def forward(self, states):
        obs = self.net(states)
        return Normal(obs, torch.ones_like(obs))


class MLPStochasticDecoder(torch.nn.Module):
    """
    decode the states to Gaussian distributions of outputs.
    """

    def __init__(
        self,
        state_dim,
        obs_dim,
        hidden_size,
        hidden_layers,
        norm=None,
        min_std=None,
        hidden_activation="elu",
    ) -> None:
        super().__init__()
        self.min_std = min_std if min_std is not None else MIN_STD
        self.mu_net = MLP(
            state_dim,
            obs_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation=hidden_activation,
        )
        self.std_net = MLP(
            state_dim,
            obs_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation=hidden_activation,
            output_activation="softplus",
        )

    def forward(self, states):
        obs_dist = Normal(self.mu_net(states), self.std_net(states) + self.min_std)
        return obs_dist


class MLPStaticStochasticDecoder(torch.nn.Module):
    """
    decode the states to Gaussian distributions of outputs with the standard deviation is a learnable global variable.
    """  # noqa: E501

    def __init__(
        self,
        state_dim,
        obs_dim,
        hidden_size,
        hidden_layers,
        norm=None,
        min_std=None,
        hidden_activation="elu",
    ) -> None:
        super().__init__()
        self.min_std = min_std if min_std is not None else MIN_STD
        self.mu_net = MLP(
            state_dim,
            obs_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation=hidden_activation,
        )
        self.log_std = nn.Parameter(torch.zeros(obs_dim))

    def forward(self, states):
        obs_dist = Normal(self.mu_net(states), torch.exp(self.log_std) + self.min_std)
        return obs_dist


class CNNDecoderHa(nn.Module):
    """
    The structure is introduced in Ha and Schmidhuber, World Model.
    NOTE: The structure only works for 64 x 64 image, pixel range [0, 1].
    """

    def __init__(self, state_dim, output_size, width=32, *args, **kwargs) -> None:
        super().__init__()
        self.latent_dim = state_dim
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 32 * width),
            nn.Unflatten(-1, (32 * width, 1, 1)),
            nn.ConvTranspose2d(32 * width, 4 * width, 5, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(4 * width, 2 * width, 5, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * width, width, 6, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(width, 3, 6, 2),
            nn.Sigmoid(),
        )

    def forward(self, state):
        head_dims = state.shape[:-1]
        state = state.view(-1, self.latent_dim)
        output = self.net(state)
        output = F.interpolate(output, self.output_size)
        return Normal(output.view(*head_dims, *output.shape[-3:]), 1)


decoder_classes = {
    "dmlp": MLPDeterministicDecoder,
    "smlp": MLPStochasticDecoder,
    "ssmlp": MLPStaticStochasticDecoder,
    "cnn_ha": CNNDecoderHa,
}


class MultimodalDecoder(nn.Module):
    def __init__(self, emb_dim, config) -> None:
        super().__init__()
        self.config = config
        self.decoders = torch.nn.ModuleDict()
        for name, dim, decoder_config in self.config:
            decoder_config = decoder_config.copy()
            decoder_type = decoder_config.pop("name")
            self.decoders[name] = decoder_classes[decoder_type](
                emb_dim, dim, **decoder_config
            )

    def forward(self, emb):
        return {name: decoder(emb).mean for name, decoder in self.decoders.items()}
