
import torch
from ray.rllib.models.torch.torch_distributions import TorchDistribution
from torch.distributions import Normal, Bernoulli

class MixedGaussianBernoulli(TorchDistribution):
    """
    Custom policy head:
    - First 2 dims -> Gaussian (movement)
    - Next 2 dims -> Bernoulli (flag1, flag2)
    - Next 3 dims -> Gaussian (power, dir_x, dir_y)
    """

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        # 2 gauss (dx,dy)
        # 2 bernoulli (flag1, flag2)
        # 3 gauss (power, dir_x, dir_y)
        # For each gaussian we output: mean + log_std = 2 params
        return 2*2 + 2 + 3*2  # = 4 + 2 + 6 = 12

    def __init__(self, inputs, model):
        super().__init__(inputs, model)

        # Interpret logits
        # Split following the defined order
        self.mean1 = inputs[:, 0:2]      # dx, dy
        self.logstd1 = inputs[:, 2:4]

        self.logits_flags = inputs[:, 4:6]  # flag1 + flag2

        self.mean2 = inputs[:, 6:9]     # power, dir_x, dir_y
        self.logstd2 = inputs[:, 9:12]

    def sample(self):
        # Movement
        std1 = torch.exp(self.logstd1)
        move = self.mean1 + std1 * torch.randn_like(std1)

        # Flags
        flag_dist = Bernoulli(logits=self.logits_flags)
        flags = flag_dist.sample()

        # Power + directions
        std2 = torch.exp(self.logstd2)
        rest = self.mean2 + std2 * torch.randn_like(std2)

        # concatenate
        return torch.cat([move, flags, rest], dim=-1)

    def logp(self, actions):
        move = actions[:, 0:2]
        flags = actions[:, 2:4]
        rest = actions[:, 4:7]

        # Movement logp
        std1 = torch.exp(self.logstd1)
        normal1 = Normal(self.mean1, std1)
        logp_move = torch.sum(normal1.log_prob(move), dim=-1)

        # Flags logp
        flag_dist = Bernoulli(logits=self.logits_flags)
        logp_flags = torch.sum(flag_dist.log_prob(flags), dim=-1)

        # Rest logp
        std2 = torch.exp(self.logstd2)
        normal2 = Normal(self.mean2, std2)
        logp_rest = torch.sum(normal2.log_prob(rest), dim=-1)

        return logp_move + logp_flags + logp_rest
