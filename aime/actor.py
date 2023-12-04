from copy import deepcopy

import numpy as np
import torch
from einops import rearrange

from aime.data import ArrayDict


class RandomActor:
    """Actor that random samples from the action space"""

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def __call__(self, obs):
        return self.action_space.sample()

    def reset(self):
        pass


class PolicyActor:
    """Model-based policy for taking actions"""

    def __init__(self, ssm, policy, eval=True) -> None:
        """
        ssm          : a state space model
        policy       : a policy take a hidden state and output the distribution of actions
        """  # noqa: E501
        self.ssm = ssm
        self.policy = policy
        self.eval = eval

    def reset(self):
        self.state = self.ssm.reset(1)
        self.model_parameter = list(self.ssm.parameters())[0]

    def __call__(self, obs):
        obs = ArrayDict(deepcopy(obs))
        obs.to_torch()
        obs.expand_dim_equal_()
        obs.to(self.model_parameter)
        obs.vmap_(lambda v: v.unsqueeze(dim=0))

        self.state, _ = self.ssm.posterior_step(obs, obs["pre_action"], self.state)
        state_feature = self.ssm.get_state_feature(self.state)
        action_dist = self.policy(state_feature)
        action = action_dist.mode if self.eval else action_dist.sample()
        action = action.detach().cpu().numpy()[0]

        return action


class StackPolicyActor:
    """Actor for the BCO policy, who needs a stack of observation to operate"""

    def __init__(self, encoder, policy, stack: int) -> None:
        self.encoder = encoder
        self.policy = policy
        self.stack = stack
        self.embs = []

    def reset(self):
        self.embs = []
        self.model_parameter = list(self.policy.parameters())[0]

    def __call__(self, obs):
        obs = ArrayDict(deepcopy(obs))
        obs.to_torch()
        obs.expand_dim_equal_()
        obs.to(self.model_parameter)
        obs.vmap_(lambda v: v.unsqueeze(dim=0))
        emb = self.encoder(obs)

        if len(self.embs) == 0:
            for _ in range(self.stack):
                self.embs.append(emb)
        else:
            self.embs.pop(0)
            self.embs.append(emb)

        emb = torch.stack(self.embs)

        emb = rearrange(emb, "t b f -> b (t f)")

        action = self.policy(emb)
        action = action.detach().cpu().numpy()[0]

        return action


class GuassianNoiseActorWrapper:
    def __init__(self, actor, noise_level, action_space) -> None:
        self._actor = actor
        self.noise_level = noise_level
        self.action_space = action_space

    def reset(self):
        return self._actor.reset()

    def __call__(self, obs):
        action = self._actor(obs)
        action = action + self.noise_level * np.random.randn(*action.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
