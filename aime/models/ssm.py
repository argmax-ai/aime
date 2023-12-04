from typing import Dict, Optional

import numpy as np
import torch
from einops import rearrange

from aime.data import ArrayDict
from aime.dist import Normal, TanhNormal, kl_divergence

from .base import MIN_STD, MLP, decoder_classes, encoder_classes
from .policy import TanhGaussianPolicy


class SSM(torch.nn.Module):
    """
    State-Space Model BaseClass.
    NOTE: in some literature, this type of model is also called sequential auto-encoder or stochastic rnn.
    NOTE: in the current version, SSM also contain encoders, decoders and probes for the
          sack of simplicity for model training. But this may not be the optimal modularization.
    """  # noqa: E501

    def __init__(
        self,
        input_config,
        output_config,
        action_dim,
        state_dim,
        probe_config=None,
        intrinsic_reward_config=None,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
        kl_scale=1.0,
        free_nats=0.0,
        kl_rebalance=None,
        nll_reweight="dim_wise",
        idm_mode="none",
        min_std=None,
        **kwargs,
    ) -> None:
        """
        input_config            : a list of tuple(name, dim, encoder_config)
        output_config           : a list of tuple(name, dim, decoder_config)
        action_dim              : int
        state_dim               : config for dims of the latent state
        probe_config            : a list of tuple(name, dim, decoder_config)
        intrinsic_reward_config : an optional config to enable the intrinsic reward based on plan2explore paper
        hidden_size             : width of the neural networks
        hidden_layers           : depth of the neural networks
        norm                    : what type of normalization layer used in the network, default is `None`.
        kl_scale                : scale for the kl term in the loss function
        free_nats               : free information per dim in latent space that is not penalized
        kl_rebalance            : rebalance the kl term with the linear combination of the two detached version, default is `None` meaning disable, enable with a float value between 0 and 1
        nll_reweight            : reweight method for the likelihood term (also the kl accordingly), choose from `none`, `modility_wise`, `dim_wise`.
        idm_mode                : mode for idm, choose from `none`, `end2end` and `detach`
        min_std                 : the minimal std for all the learnable distributions for numerical stablity, set to None will follow the global default.

        NOTE: For output and probe configs, there can be a special name `emb` which indicate to predict the detached embedding from the encoders.
              For that use case, the `dim` in that config tuple will be overwrite.
        """  # noqa: E501
        super().__init__()
        self.input_config = input_config
        self.output_config = output_config
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.probe_config = probe_config
        self.intrinsic_reward_config = intrinsic_reward_config
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.norm = norm
        self.kl_scale = kl_scale
        self.free_nats = free_nats
        self.kl_rebalance = kl_rebalance
        self.nll_reweight = nll_reweight
        assert self.nll_reweight in ("none", "modility_wise", "dim_wise")
        self.idm_mode = idm_mode
        assert self.idm_mode in (
            "none",
            "end2end",
            "detach",
        ), f"recieved unknown idm_mode `{self.idm_mode}`."
        self.min_std = min_std if min_std is not None else MIN_STD

        self.create_network()
        self.metric_func = torch.nn.MSELoss()

    def create_network(
        self,
    ):
        self.encoders = torch.nn.ModuleDict()
        for name, dim, encoder_config in self.input_config:
            encoder_config = encoder_config.copy()
            encoder_type = encoder_config.pop("name")
            self.encoders[name] = encoder_classes[encoder_type](dim, **encoder_config)

        self.emb_dim = sum(
            [encoder.output_dim for name, encoder in self.encoders.items()]
        )

        self.decoders = torch.nn.ModuleDict()
        for name, dim, decoder_config in self.output_config:
            if name == "emb":
                dim = self.emb_dim
            decoder_config = decoder_config.copy()
            decoder_type = decoder_config.pop("name")
            self.decoders[name] = decoder_classes[decoder_type](
                self.state_feature_dim, dim, **decoder_config
            )

        self.probes = torch.nn.ModuleDict()
        for name, dim, decoder_config in self.probe_config:
            if name == "emb":
                dim = self.emb_dim
            decoder_config = decoder_config.copy()
            decoder_type = decoder_config.pop("name")
            self.probes[name] = decoder_classes[decoder_type](
                self.state_feature_dim, dim, **decoder_config
            )

        if not (self.idm_mode == "none"):
            # Inverse Dynamic Model (IDM) is a non-casual policy
            self.idm = TanhGaussianPolicy(
                self.state_feature_dim + self.emb_dim,
                self.action_dim,
                self.hidden_size,
                self.hidden_layers,
            )

        if self.intrinsic_reward_config is not None:
            self.emb_prediction_heads = torch.nn.ModuleList(
                [
                    MLP(
                        self.state_feature_dim + self.action_dim,
                        self.emb_dim,
                        self.intrinsic_reward_config["hidden_size"],
                        self.intrinsic_reward_config["hidden_layers"],
                    )
                    for _ in range(self.intrinsic_reward_config["num_ensembles"])
                ]
            )

    def reset(self, batch_size):
        """reset the hidden state of the SSM"""
        raise NotImplementedError

    @property
    def state_feature_dim(self):
        return self.state_dim

    def stack_states(self, states):
        return ArrayDict.stack(states, dim=0)

    def flatten_states(self, states):
        # flatten the sequence of states as the starting state of rollout
        if isinstance(states, list):
            states = self.stack_states(states)
        states.vmap_(lambda v: rearrange(v, "t b f -> (t b) f"))
        return states

    def get_state_feature(self, state):
        return state

    def get_emb(self, obs):
        return torch.cat(
            [model(obs[name]) for name, model in self.encoders.items()], dim=-1
        )

    def get_output_dists(self, state_feature, names=None):
        if names is None:
            names = self.decoders.keys()
        return {name: self.decoders[name](state_feature) for name in names}

    def get_outputs(self, state_feature, names=None):
        if names is None:
            names = self.decoders.keys()
        return {name: self.decoders[name](state_feature).mode for name in names}

    def get_probe_dists(self, state_feature, names=None):
        if names is None:
            names = self.probes.keys()
        return {name: self.probes[name](state_feature) for name in names}

    def get_probes(self, state_feature, names=None):
        if names is None:
            names = self.probes.keys()
        return {name: self.probes[name](state_feature).mode for name in names}

    def compute_kl(self, posterior, prior):
        if self.kl_rebalance is None:
            return kl_divergence(posterior, prior)
        else:
            return self.kl_rebalance * kl_divergence(posterior.detach(), prior) + (
                1 - self.kl_rebalance
            ) * kl_divergence(posterior, prior.detach())

    def forward(self, obs_seq, pre_action_seq, filter_step=None):
        """the call for training the model"""
        if filter_step is None:
            filter_step = len(obs_seq)
        state = self.reset(obs_seq[self.input_config[0][0]].shape[1])
        emb_seq = self.get_emb(obs_seq)
        obs_seq["emb"] = emb_seq.detach()

        states, kls = self.filter(
            obs_seq[:filter_step],
            pre_action_seq[:filter_step],
            emb_seq[:filter_step],
            state,
        )
        states = states + self.rollout(states[-1], pre_action_seq[filter_step:])

        # clamp the kls with free nats, but keep the real value at log
        clamp_kls = (
            torch.clamp_min(torch.sum(kls, dim=-1, keepdim=True), self.free_nats)
            / kls.shape[-1]
        )
        kls = clamp_kls + (kls - clamp_kls).detach()

        state_features = torch.stack(
            [self.get_state_feature(state) for state in states]
        )
        output_dists = self.get_output_dists(state_features)

        metrics = {}
        rec_term = 0
        for name, dist in output_dists.items():
            _r_term = torch.mean(
                torch.flatten(dist.log_prob(obs_seq[name]), 2).sum(dim=-1).sum(dim=0)
            )
            rec_term = rec_term + _r_term
            metrics[f"{name}_mse"] = self.metric_func(dist.mean, obs_seq[name]).item()
            metrics[f"{name}_rec_term"] = _r_term.item()
        kl_term = -torch.mean(kls.sum(dim=-1).sum(dim=0))
        elbo = rec_term + kl_term
        metrics.update(
            {
                "rec_term": rec_term.item(),
                "kl_term": kl_term.item(),
                "elbo": elbo.item(),
            }
        )

        reconstruction_loss = 0
        kl_loss = 0

        if self.nll_reweight == "none":
            reconstruction_loss = -rec_term
            kl_loss = -kl_term
        # Reference: Seitzer et. al., On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks, ICLR 2022  # noqa: E501
        # NOTE: the original version only reweight the log_prob, but here I think if the likelihood is reweighted, the kl should be reweighted accordingly.  # noqa: E501
        elif self.nll_reweight == "modility_wise":
            for name, dist in output_dists.items():
                _r_loss = -torch.mean(
                    torch.flatten(
                        dist.log_prob(obs_seq[name]) * dist.stddev.detach(), 2
                    )
                    .sum(dim=-1)
                    .sum(dim=0)
                )
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_mse"] = self.metric_func(
                    dist.mean, obs_seq[name]
                ).item()
                metrics[f"{name}_reconstruction_loss"] = _r_loss.item()
                kl_loss = kl_loss + torch.mean(
                    (
                        kls.sum(dim=-1)
                        * torch.flatten(dist.stddev[: kls.shape[0]], 2)
                        .detach()
                        .mean(dim=-1)
                    ).sum(dim=0)
                )
            kl_loss = kl_loss / len(output_dists)
        elif self.nll_reweight == "dim_wise":
            total_dims = 0
            for name, dist in output_dists.items():
                _r_loss = -torch.mean(
                    torch.flatten(
                        dist.log_prob(obs_seq[name]) * dist.stddev.detach(), 2
                    )
                    .sum(dim=-1)
                    .sum(dim=0)
                )
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_mse"] = self.metric_func(
                    dist.mean, obs_seq[name]
                ).item()
                metrics[f"{name}_reconstruction_loss"] = _r_loss.item()
                kl_loss = kl_loss + torch.mean(
                    (
                        kls.sum(dim=-1, keepdim=True)
                        * torch.flatten(dist.stddev[: kls.shape[0]], 2).detach()
                    )
                    .sum(dim=-1)
                    .sum(dim=0)
                )
                total_dims = total_dims + np.prod(dist.stddev.shape[2:])
            kl_loss = kl_loss / total_dims

        if self.probe_config is not None:
            # ad hoc skip the first state because there is no initial state estimator
            probe_state_features = state_features.detach()[1:]
            probe_dists = self.get_probe_dists(probe_state_features)
            for name, dist in probe_dists.items():
                _r_loss = -torch.mean(
                    dist.log_prob(obs_seq[name][1:]).sum(dim=-1).sum(dim=0)
                )
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_probe_mse"] = self.metric_func(
                    dist.mean, obs_seq[name][1:]
                ).item()
                metrics[f"{name}_probe_reconstruction_loss"] = _r_loss.item()

        loss = reconstruction_loss + self.kl_scale * kl_loss

        if not (self.idm_mode == "none" or self.idm is None):
            idm_inputs = torch.cat([state_features[:-1], emb_seq[1:]], dim=-1)
            if self.idm_mode == "detach":
                idm_inputs = idm_inputs.detach()
            action_dist = self.idm(idm_inputs)
            idm_loss = (
                -action_dist.log_prob(pre_action_seq[1:]).sum(dim=-1).mean(dim=0).sum()
            )
            idm_mse = self.metric_func(action_dist.mode, pre_action_seq[1:])

            metrics.update({"idm_loss": idm_loss.item(), "idm_mse": idm_mse.item()})

            loss = loss + idm_loss

        if self.intrinsic_reward_config is not None:
            inputs = torch.cat(
                [state_features[:-1], obs_seq["pre_action"][1:]], dim=-1
            ).detach()
            outputs = [head(inputs) for head in self.emb_prediction_heads]
            target = emb_seq[1:].detach()
            emb_prediction_loss = sum(
                [self.metric_func(output, target) for output in outputs]
            )
            loss = loss + emb_prediction_loss
            metrics["emb_prediction_loss"] = emb_prediction_loss.item()

        metrics.update(
            {
                "total_loss": loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "kl_loss": kl_loss.item(),
            }
        )

        return (
            ArrayDict({name: dist.mean for name, dist in output_dists.items()}),
            states,
            loss,
            metrics,
        )

    def filter(self, obs_seq, pre_action_seq, emb_seq=None, initial_state=None):
        if initial_state is None:
            initial_state = self.reset(obs_seq[self.input_config[0][0]].shape[1])
        if emb_seq is None:
            emb_seq = self.get_emb(obs_seq)
        assert pre_action_seq.shape[0] == emb_seq.shape[0]
        states = []
        kls = []
        state = initial_state
        for t in range(emb_seq.shape[0]):
            state, kl = self.posterior_step(None, pre_action_seq[t], state, emb_seq[t])
            states.append(state)
            kls.append(kl)
        kls = torch.stack(kls, dim=0)
        return states, kls

    def rollout(self, initial_state, action_seq):
        state = initial_state
        states = []
        for t in range(action_seq.shape[0]):
            state = self.prior_step(action_seq[t], state)
            states.append(state)
        return states

    def posterior_step(self, obs, pre_action, state, emb=None, determinastic=False):
        raise NotImplementedError

    def prior_step(self, pre_action, state, determinastic=False):
        raise NotImplementedError

    def generate(self, initial_state, action_seq, names=None):
        states = self.rollout(initial_state, action_seq)

        state_features = torch.stack(
            [self.get_state_feature(state) for state in states]
        )

        return states, ArrayDict(self.get_outputs(state_features, names))

    def filter_with_policy(
        self, obs_seq, policy, idm=None, filter_step=None, kl_only=False
    ):
        """
        Filter the states with a policy that generate actions.
        This is used for the AIME algorithm for now.
        """
        if filter_step is None:
            filter_step = len(obs_seq)
        state = self.reset(obs_seq[self.input_config[0][0]].shape[1])
        emb_seq = self.get_emb(obs_seq)

        states = []
        kls = []
        actions_kls = []
        actions = []

        for t in range(filter_step):
            if idm is None:
                action_dist = policy(self.get_state_feature(state))
                action = action_dist.rsample()
            else:
                state_feature = self.get_state_feature(state)
                action_posterior_dist = idm(
                    torch.cat([state_feature, emb_seq[t]], dim=-1)
                )
                action_prior_dist = policy(state_feature)
                actions_kl = torch.distributions.kl_divergence(
                    action_posterior_dist, action_prior_dist
                )
                actions_kls.append(actions_kl)
                action = action_posterior_dist.rsample()
            state, kl = self.posterior_step(None, action, state, emb_seq[t])
            states.append(state)
            kls.append(kl)
            actions.append(action)

        for t in range(filter_step, len(obs_seq)):
            action_dist = policy(self.get_state_feature(state))
            action = action_dist.rsample()
            state = self.prior_step(action, state)
            states.append(state)
            actions.append(action)

        kls = torch.stack(kls)
        if idm is not None:
            action_kls = torch.stack(actions_kls)
        actions = torch.stack(actions)

        state_features = torch.stack(
            [self.get_state_feature(state) for state in states]
        )
        output_dists = self.get_output_dists(state_features)

        metrics = {}
        rec_term = 0
        for name, dist in output_dists.items():
            _r_term = torch.mean(
                torch.flatten(dist.log_prob(obs_seq[name]), 2).sum(dim=-1).sum(dim=0)
            )
            rec_term = rec_term + _r_term
            metrics[f"{name}_mse"] = self.metric_func(dist.mean, obs_seq[name]).item()
            metrics[f"{name}_rec_term"] = _r_term.item()
        kl_term = -torch.mean(kls.sum(dim=-1).sum(dim=0))
        elbo = rec_term + kl_term
        metrics.update(
            {
                "rec_term": rec_term.item(),
                "kl_term": kl_term.item(),
                "elbo": elbo.item(),
            }
        )

        reconstruction_loss = 0
        kl_loss = 0

        if self.nll_reweight == "none":
            reconstruction_loss = -rec_term
            kl_loss = -kl_term
        # Reference: Seitzer et. al., On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks, ICLR 2022  # noqa: E501
        # NOTE: the original version only reweight the log_prob, but here I think if the likelihood is reweighted, the kl should be reweighted accordingly.  # noqa: E501
        elif self.nll_reweight == "modility_wise":
            for name, dist in output_dists.items():
                _r_loss = -torch.mean(
                    torch.flatten(
                        dist.log_prob(obs_seq[name]) * dist.stddev.detach(), 2
                    )
                    .sum(dim=-1)
                    .sum(dim=0)
                )
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_mse"] = self.metric_func(
                    dist.mean, obs_seq[name]
                ).item()
                metrics[f"{name}_reconstruction_loss"] = _r_loss.item()
                kl_loss = kl_loss + torch.mean(
                    (
                        kls.sum(dim=-1)
                        * torch.flatten(dist.stddev[: kls.shape[0]], 2)
                        .detach()
                        .mean(dim=-1)
                    ).sum(dim=0)
                )
            kl_loss = kl_loss / len(output_dists)
        elif self.nll_reweight == "dim_wise":
            total_dims = 0
            for name, dist in output_dists.items():
                _r_loss = -torch.mean(
                    torch.flatten(
                        dist.log_prob(obs_seq[name]) * dist.stddev.detach(), 2
                    )
                    .sum(dim=-1)
                    .sum(dim=0)
                )
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_mse"] = self.metric_func(
                    dist.mean, obs_seq[name]
                ).item()
                metrics[f"{name}_reconstruction_loss"] = _r_loss.item()
                kl_loss = kl_loss + torch.mean(
                    (
                        kls.sum(dim=-1, keepdim=True)
                        * torch.flatten(dist.stddev[: kls.shape[0]], 2).detach()
                    )
                    .sum(dim=-1)
                    .sum(dim=0)
                )
                total_dims = total_dims + np.prod(dist.stddev.shape[2:])
            kl_loss = kl_loss / total_dims

        if self.probe_config is not None:
            # ad hoc skip the first state because there is no initial state estimator
            probe_state_features = state_features.detach()[1:]
            probe_dists = self.get_probe_dists(probe_state_features)
            for name, dist in probe_dists.items():
                _r_loss = -torch.mean(
                    dist.log_prob(obs_seq[name][1:]).sum(dim=-1).sum(dim=0)
                )
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_probe_mse"] = self.metric_func(
                    dist.mean, obs_seq[name][1:]
                ).item()
                metrics[f"{name}_probe_reconstruction_loss"] = _r_loss.item()

        if not kl_only:
            loss = reconstruction_loss + self.kl_scale * kl_loss
        else:
            loss = self.kl_scale * kl_loss

        if idm is not None:
            action_kl_loss = action_kls.sum(dim=-1).sum(dim=0).mean()
            loss = loss + self.kl_scale * action_kl_loss
            metrics["action_kl_loss"] = action_kl_loss.item()

        metrics.update(
            {
                "total_loss": loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "model_kl_loss": kl_loss.item(),
            }
        )

        return (
            ArrayDict({name: dist.mean for name, dist in output_dists.items()}),
            states,
            actions,
            loss,
            metrics,
        )

    def rollout_with_policy(
        self,
        initial_state,
        policy,
        horizon,
        names=None,
        state_detach=False,
        action_sample=True,
    ):
        """
        Rollout the world model in imagination with a policy.
        In this way, the world model serves as a virtual environment for the agent.
        This is used for Dyna-style algorithm like Dreamer.
        """
        state = initial_state
        states = []
        actions = []
        action_entropys = []
        for t in range(horizon):
            state_feature = self.get_state_feature(state)
            if state_detach:
                state_feature = state_feature.detach()
            action_dist = policy(state_feature)
            action = action_dist.rsample() if action_sample else action_dist.mode
            actions.append(action)
            # the empirical maximal entropy tanhNormal is (0, 0.8742) in which p(0.5) = 0.5  # noqa: E501
            action_entropys.append(kl_divergence(action_dist, TanhNormal(0, 0.88)))
            state = self.prior_step(action, state)
            states.append(state)

        state_features = torch.stack(
            [self.get_state_feature(state) for state in states]
        )
        actions = torch.stack(actions, dim=0)
        action_entropys = torch.stack(action_entropys, dim=0)

        outputs = ArrayDict(self.get_outputs(state_features, names))

        if self.intrinsic_reward_config is not None:
            initial_state_feature = self.get_state_feature(initial_state).unsqueeze(
                dim=0
            )

            inputs = torch.cat(
                [
                    torch.cat([initial_state_feature, state_features[:-1]], dim=0),
                    actions,
                ],
                dim=-1,
            )

            emb_predictions = [head(inputs) for head in self.emb_prediction_heads]
            emb_predictions = torch.stack(emb_predictions, dim=0)
            disagreements = torch.var(emb_predictions, dim=0)
            intrinsic_rewards = torch.mean(disagreements.log(), dim=-1, keepdim=True)
            outputs["intrinsic_reward"] = intrinsic_rewards

        outputs["action_entropy"] = action_entropys

        return states, actions, outputs


class RSSM(SSM):
    """
    The one from PlaNet and Dreamer, name is short for Recurrent State-Space Model.
    Interpretation of the latent variables:
        h: history of all the states before (exclude the current time step)
        s: information about the current time step (ambiguity reduced by the history)
    """

    def create_network(self):
        super().create_network()

        memory_dim, state_dim = self.state_dim

        self.prior_mean = MLP(
            memory_dim, state_dim, self.hidden_size, self.hidden_layers, self.norm
        )
        self.prior_std = MLP(
            memory_dim,
            state_dim,
            self.hidden_size,
            self.hidden_layers,
            self.norm,
            output_activation="softplus",
        )
        self.posterior_mean = MLP(
            self.emb_dim + memory_dim,
            state_dim,
            self.hidden_size,
            self.hidden_layers,
            self.norm,
        )
        self.posterior_std = MLP(
            self.emb_dim + memory_dim,
            state_dim,
            self.hidden_size,
            self.hidden_layers,
            self.norm,
            output_activation="softplus",
        )
        self.memory_cell = torch.nn.GRUCell(state_dim + self.action_dim, memory_dim)

        self.register_buffer("initial_memory", torch.zeros(1, memory_dim))
        self.register_buffer("initial_state", torch.zeros(1, state_dim))

    @property
    def state_feature_dim(self) -> int:
        return sum(self.state_dim)

    def get_state_feature(self, state: ArrayDict) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def reset(self, batch_size: int) -> ArrayDict:
        return ArrayDict(
            deter=torch.repeat_interleave(self.initial_memory, batch_size, dim=0),
            stoch=torch.repeat_interleave(self.initial_state, batch_size, dim=0),
        )

    def posterior_step(
        self,
        obs: ArrayDict,
        pre_action: torch.Tensor,
        state: ArrayDict,
        emb: Optional[torch.Tensor] = None,
        determinastic=False,
    ):
        h, s = state["deter"], state["stoch"]

        if emb is None:
            emb = self.get_emb(obs)

        # 1. update the determinastic part
        h = self.memory_cell(torch.cat([s, pre_action], dim=-1), h)

        # 2. compute the prior
        prior = Normal(self.prior_mean(h), self.prior_std(h) + self.min_std)

        # 3. compute the posterior
        info = torch.cat([h, emb], dim=-1)
        posterior = Normal(
            self.posterior_mean(info), self.posterior_std(info) + self.min_std
        )

        # 4. determine the state
        s = posterior.rsample() if not determinastic else posterior.mean

        # 5. compute kl for loss
        kl = self.compute_kl(posterior, prior)

        return ArrayDict(deter=h, stoch=s), kl

    def prior_step(self, pre_action, state, determinastic=False):
        h, s = state["deter"], state["stoch"]

        # 1. update the determinastic part
        h = self.memory_cell(torch.cat([s, pre_action], dim=-1), h)

        # 2. compute the prior
        prior = Normal(self.prior_mean(h), self.prior_std(h) + self.min_std)

        # 3. update the stochastic part
        s = prior.rsample() if not determinastic else prior.mean

        return ArrayDict(deter=h, stoch=s)


class RSSMO(RSSM):
    """
    Implement everything in the same way as the original repo.
    """

    def create_network(self):
        # NOTE: this temprary implementation only consider visual and tabular modilities
        self.encoders = torch.nn.ModuleDict()
        self.modilities_and_keys = {"visual": [], "tabular": []}
        tabular_dims = 0
        for name, dim, encoder_config in self.input_config:
            encoder_config = encoder_config.copy()
            encoder_type = encoder_config.pop("name")
            if encoder_type == "mlp":
                self.modilities_and_keys["tabular"].append(name)
                tabular_encoder_config = encoder_config
                tabular_dims += dim
            else:
                self.modilities_and_keys["visual"].append(name)
                self.encoders[name] = encoder_classes[encoder_type](
                    dim, **encoder_config
                )

        if tabular_dims > 0:
            self.encoders["tabular"] = encoder_classes["mlp"](
                tabular_dims, **tabular_encoder_config
            )

        self.emb_dim = sum(
            [encoder.output_dim for name, encoder in self.encoders.items()]
        )

        self.decoders = torch.nn.ModuleDict()
        for name, dim, decoder_config in self.output_config:
            if name == "emb":
                dim = self.emb_dim
            decoder_config = decoder_config.copy()
            decoder_type = decoder_config.pop("name")
            self.decoders[name] = decoder_classes[decoder_type](
                self.state_feature_dim, dim, **decoder_config
            )

        self.probes = torch.nn.ModuleDict()
        for name, dim, decoder_config in self.probe_config:
            if name == "emb":
                dim = self.emb_dim
            decoder_config = decoder_config.copy()
            decoder_type = decoder_config.pop("name")
            self.probes[name] = decoder_classes[decoder_type](
                self.state_feature_dim, dim, **decoder_config
            )

        if not (self.idm_mode == "none"):
            # Inverse Dynamic Model (IDM) is a non-casual policy
            self.idm = TanhGaussianPolicy(
                self.state_feature_dim + self.emb_dim,
                self.action_dim,
                self.hidden_size,
                self.hidden_layers,
            )

        if self.intrinsic_reward_config is not None:
            self.emb_prediction_heads = torch.nn.ModuleList(
                [
                    MLP(
                        self.state_feature_dim + self.action_dim,
                        self.emb_dim,
                        self.intrinsic_reward_config["hidden_size"],
                        self.intrinsic_reward_config["hidden_layers"],
                    )
                    for _ in range(self.intrinsic_reward_config["num_ensembles"])
                ]
            )

        memory_dim, state_dim = self.state_dim

        self.pre_memory = MLP(
            state_dim + self.action_dim,
            None,
            self.hidden_size,
            1,
            self.norm,
            have_head=False,
            hidden_activation="swish",
        )
        self.memory_cell = torch.nn.GRUCell(self.hidden_size, memory_dim)

        self.prior_net = MLP(
            memory_dim,
            2 * state_dim,
            self.hidden_size,
            1,
            self.norm,
            hidden_activation="swish",
        )
        self.posterior_net = MLP(
            self.emb_dim + memory_dim,
            2 * state_dim,
            self.hidden_size,
            1,
            self.norm,
            hidden_activation="swish",
        )

        self.register_buffer("initial_memory", torch.zeros(1, memory_dim))
        self.register_buffer("initial_state", torch.zeros(1, state_dim))

    def get_emb(self, obs):
        outputs = []
        for name, model in self.encoders.items():
            if name == "tabular":
                inputs = torch.cat(
                    [obs[name] for name in self.modilities_and_keys["tabular"]], dim=-1
                )
            else:
                inputs = obs[name]
            outputs.append(model(inputs))
        return torch.cat(outputs, dim=-1)

    def posterior_step(
        self,
        obs: ArrayDict,
        pre_action: torch.Tensor,
        state: ArrayDict,
        emb: Optional[torch.Tensor] = None,
        determinastic=False,
    ):
        h, s = state["deter"], state["stoch"]

        if emb is None:
            emb = self.get_emb(obs)

        # 1. update the determinastic part
        info = self.pre_memory(torch.cat([s, pre_action], dim=-1))
        h = self.memory_cell(info, h)

        # 2. compute the prior
        prior_mean, prior_std = torch.chunk(self.prior_net(h), 2, dim=-1)
        prior_std = self._sigmoid2(prior_std) + self.min_std
        prior = Normal(prior_mean, prior_std)

        # 3. compute the posterior
        info = torch.cat([h, emb], dim=-1)
        posterior_mean, posterior_std = torch.chunk(self.posterior_net(info), 2, dim=-1)
        posterior_std = self._sigmoid2(posterior_std) + self.min_std
        posterior = Normal(posterior_mean, posterior_std)

        # 4. determine the state
        s = posterior.rsample() if not determinastic else posterior.mean

        # 5. compute kl for loss
        kl = self.compute_kl(posterior, prior)

        return ArrayDict(deter=h, stoch=s), kl

    def prior_step(self, pre_action, state, determinastic=False):
        h, s = state["deter"], state["stoch"]

        # 1. update the determinastic part
        info = self.pre_memory(torch.cat([s, pre_action], dim=-1))
        h = self.memory_cell(info, h)

        # 2. compute the prior
        prior_mean, prior_std = torch.chunk(self.prior_net(h), 2, dim=-1)
        prior_std = self._sigmoid2(prior_std) + self.min_std
        prior = Normal(prior_mean, prior_std)

        # 3. update the stochastic part
        s = prior.rsample() if not determinastic else prior.mean

        return ArrayDict(deter=h, stoch=s)

    def _sigmoid2(self, x):
        # from dreamerv2
        return 2 * torch.sigmoid(x / 2)


ssm_classes: Dict[str, SSM] = {
    "rssm": RSSM,
    "rssmo": RSSMO,
}
