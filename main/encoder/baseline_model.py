import torch
from torch.nn.modules.linear import Linear
from modules import *
from utils import *
import torch.nn as nn
import numpy as np
from transformer import ShallowTransformer, DeepTransformer

class BaselineModel(nn.Module):
    def __init__(
        self,
        dist_size=None, 
        seq_len=1000, 
        state_space=4,
        action_space=2,
        key_hidden_size=256,
        value_hidden_size=256
    ):
        super(BaselineModel, self).__init__()
        self.input_size = state_space + action_space
        self.dist_size = dist_size if dist_size is not None else 10
        self.linear1 = LinearLayer(input_size=self.input_size, output_size=int(key_hidden_size/2))
        self.linear2 = LinearLayer(input_size=int(key_hidden_size/2), output_size=key_hidden_size)
        self.linear3 = LinearLayer(input_size=key_hidden_size, output_size=key_hidden_size)
        self.transformer = DeepTransformer(
            key_hidden_size,
            seq_len,
            key_hidden_size=key_hidden_size, 
            value_hidden_size=value_hidden_size,
            num_blocks=1
        )
        self.compact_last = LinearLayer(input_size=key_hidden_size, output_size=self.dist_size*2)
        self.mu_layer = LinearLayer(input_size=key_hidden_size, output_size=self.dist_size)
        self.logvar_layer = LinearLayer(input_size=key_hidden_size, output_size=self.dist_size)

    # forward for reconstruction
    def forward(self, states, actions):
        # Example expert data
        # experts["states"] -> [batch_size, traj_length, state_space]
        # experts["actions"] -> [batch_size, traj_length, action_space]
        # states = experts["states"]  # Shape: [batch_size, traj_length, state_space]
        # actions = experts["actions"]  # Shape: [batch_size, traj_length, action_space]

        # Concatenate each state's timestep with the corresponding action's timestep
        # Resulting shape will be [batch_size, traj_length, state_space + action_space]
        state_action_concat = torch.cat((states, actions), dim=-1)

        # MLP to encode the state-action pair
        out = self.linear1(state_action_concat)
        out = self.linear2(out)
        out = self.linear3(out)
        
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        transformed = self.transformer(out, None)
        # last_token = transformed[:, -1, :] # 1 1000 256 -> 1 256
        last_token = transformed 
        # 1 256
        # compacted = self.compact_last(last_token)
        # # 1 128
        # self.mean = compacted[:, :self.dist_size] # 64
        # self.std = compacted[:, self.dist_size:] # 64
        self.mean = self.mu_layer(last_token).squeeze(0)
        self.std = self.logvar_layer(last_token).squeeze(0)
        return self.mean, self.std

    def get_dist(self):
        def reparameterize(mu, logvar):
            """
            Reparameterization trick to sample from N(mu, var) from
            N(0,1).
            :param mu: (Tensor) Mean of the latent Gaussian [B x D]
            :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
            :return: (Tensor) [B x D]
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        cond = reparameterize(self.mean, self.std)
        cond = cond.squeeze(0)
        return cond