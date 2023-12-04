import torch
from torch.distributions import TransformedDistribution
from torch.distributions.kl import kl_divergence, register_kl
from torch.distributions.transforms import TanhTransform

TANH_CLIP = 0.999
NUM_KL_APPROXIMATE_SAMPLES = 1024


class Normal(torch.distributions.Normal):
    @property  # make pytorch < 1.12 compatible with the mode api
    def mode(self):
        return self.mean

    def detach(self):
        """return a new distribution with the same parameters but the gradients are detached"""  # noqa: E501
        return Normal(self.mean.detach(), self.scale.detach())


class TanhNormal(torch.distributions.Distribution):
    def __init__(self, mean, std, validate_args=False):
        self.base = Normal(mean, std, validate_args)
        super().__init__(self.base.batch_shape, self.base.event_shape, validate_args)
        self.dist = TransformedDistribution(self.base, TanhTransform(), validate_args)

    def __getattr__(self, name):
        return getattr(self.dist, name)

    def rsample(self, sample_shape=torch.Size()):
        return self.dist.rsample(sample_shape)

    def log_prob(self, value):
        # NOTE: normally, we don't need gradient from here
        value = torch.clamp(value, -TANH_CLIP, TANH_CLIP)
        return self.base.log_prob(torch.atanh(value)) - torch.log(1 - value**2)

    @property
    def mode(self):
        """NOTE: this is not really the mode, just a easy computation"""
        return torch.tanh(self.base.mode)

    def detach(self):
        """return a new distribution with the same parameters but the gradients are detached"""  # noqa: E501
        return TanhNormal(self.base.mean.detach(), self.base.scale.detach())


@register_kl(TanhNormal, TanhNormal)
def _kl_tanhnormal_tanhnormal(p: TanhNormal, q: TanhNormal):
    # NOTE: kl between two distribution transformed with the same transformation,
    #       is equal to the kl between the two distribution before transformation.
    return kl_divergence(p.base, q.base)


@register_kl(Normal, TanhNormal)
def _kl_normal_tanhnormal(p: Normal, q: TanhNormal):
    # NOTE: This quantity should be infinity in theory due to the fact that
    #       Noraml cover space that is not covered by TanhNormal.
    #       Here the quantity is fakely computed just to fit in the equation.
    samples = p.sample((NUM_KL_APPROXIMATE_SAMPLES,))
    logp = p.entropy()
    logq = q.log_prob(samples)
    return logp - logq.mean(dim=0)


@register_kl(TanhNormal, Normal)
def _kl_tanhnormal_normal(p: TanhNormal, q: Normal):
    samples = p.sample((NUM_KL_APPROXIMATE_SAMPLES,))
    logp = p.log_prob(samples)
    logq = q.log_prob(samples)
    return (logp - logq).mean(dim=0)
