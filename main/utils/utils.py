import os
import pickle
import random
from typing import IO, Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def evaluate(actor, env, conditions, num_episodes=10, vis=True, cond_dim=10, cond_type="random", eval_index=0, experimental="none"):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = []
    total_returns = []

    while len(total_returns) < num_episodes:
        state = env.reset()
        done = False
        terminated = False
        cond = get_random_cond(cond_dim, cond_type, conditions, eval_index=eval_index, experimental=experimental)
        with eval_mode(actor):
            while not done and not terminated:
                if "kitchen" in env.spec.id:
                    state = state[:30] # HACK: GCPC kitchen dataset has observation of length 30
                if cond_type=="none":
                    action = actor.choose_action(state, sample=False)
                else:
                    action = actor.choose_action((state,cond), sample=False)
                next_state, reward, done, info = env.step(action)
                state = next_state

                if 'episode' in info.keys():
                    total_returns.append(info['episode']['r'])
                    total_timesteps.append(info['episode']['l'])

    return total_returns, total_timesteps

def get_random_cond(cond_dim, cond_type, conditions, eval_index=0, experimental="none"):
    conds = conditions["emb"]
    # select random index from conds length
    index = random.randint(0, len(conds)-1)
    if "weighted_lowAndHigh" in experimental:
        if experimental=="weighted_lowAndHigh":
            alpha = 0.5
        else:
            # Extract the alpha value from the experimental string
            alpha_str = experimental.replace("weighted_lowAndHigh", "")
            try:
                alpha = float(alpha_str)  # Convert the string to a float
            except ValueError:
                raise ValueError(f"Invalid alpha value extracted: '{alpha_str}'")

        # Ensure alpha is within the valid range [0, 1]
        # if not (0 <= alpha <= 1):
        #     raise ValueError(f"Alpha value must be between 0 and 1. Got: {alpha}")

        eval_index=0
        # Calculate the average of the current condition and the condition with index+10
        cond_first = conds[eval_index][:cond_dim]
        cond_second = conds[(eval_index+20) % len(conds)][:cond_dim]
        cond = alpha * cond_first + (1 - alpha) * cond_second
    elif "weighted_lowAndMedium" in experimental:
        if experimental=="weighted_lowAndMedium":
            alpha = 0.5
        else:
            # Extract the alpha value from the experimental string
            alpha_str = experimental.replace("weighted_lowAndMedium", "")
            try:
                alpha = float(alpha_str)  # Convert the string to a float
            except ValueError:
                raise ValueError(f"Invalid alpha value extracted: '{alpha_str}'")

        eval_index=0
        # Calculate the average of the current condition and the condition with index+10
        cond_first = conds[eval_index][:cond_dim]
        cond_second = conds[(eval_index+10) % len(conds)][:cond_dim]
        cond = alpha * cond_first + (1 - alpha) * cond_second
    elif "weighted_mediumAndHigh" in experimental:
        if experimental=="weighted_mediumAndHigh":
            alpha = 0.5
        else:
            # Extract the alpha value from the experimental string
            alpha_str = experimental.replace("weighted_mediumAndHigh", "")
            try:
                alpha = float(alpha_str)  # Convert the string to a float
            except ValueError:
                raise ValueError(f"Invalid alpha value extracted: '{alpha_str}'")

        eval_index=10
        # Calculate the average of the current condition and the condition with index+10
        cond_first = conds[eval_index][:cond_dim]
        cond_second = conds[(eval_index+10) % len(conds)][:cond_dim]
        cond = alpha * cond_first + (1 - alpha) * cond_second
    elif experimental=="noise":
        # modify one dim of the condition
        cond = conds[eval_index][:cond_dim]
        cond[random.randint(0,cond_dim-1)] = 0
    elif "perturb" in experimental:
        try:
            perturb_index_str, perturb_value_str = experimental.replace("perturb", "").split("_")
            perturb_index = int(perturb_index_str)  # Convert the index to an integer
            perturb_value = float(perturb_value_str)  # Convert the value to a float
        except (ValueError, IndexError):
            raise ValueError(f"Invalid experimental format: '{experimental}'")

        eval_index = 20
        cond = conds[eval_index][:cond_dim]
        cond[perturb_index] = perturb_value

    elif cond_type=="random":
        cond = conds[index][:cond_dim]
    elif cond_type=="debug":
        cond = conds[eval_index][:cond_dim]
    elif cond_type=="none" or cond_type=="dummy":
        cond = [-1]*cond_dim
    else:
        # throw error that cond_type is not recognized
        raise ValueError("cond_type is not recognized. Use 'random', 'debug', 'dummy', or 'none'")
    return cond

def read_file(path: str, file_handle: IO[Any]) -> Dict[str, Any]:
    """Read file from the input path. Assumes the file stores dictionary data.

    Args:
        path:               Local or S3 file path.
        file_handle:        File handle for file.

    Returns:
        The dictionary representation of the file.
    """
    if path.endswith("pt"):
        data = torch.load(file_handle)
    elif path.endswith("pkl"):
        data = pickle.load(file_handle)
    elif path.endswith("npy"):
        data = np.load(file_handle, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
    else:
        raise NotImplementedError
    return data

def weighted_softmax(x, weights):
    x = x - torch.max(x, dim=0)[0]
    return weights * torch.exp(x) / torch.sum(
        weights * torch.exp(x), dim=0, keepdim=True)


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def get_concat_samples(policy_batch, expert_batch, args):
    # online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch

    # expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = expert_batch
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done, online_batch_cond, online_batch_dist_params = policy_batch
    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done, expert_batch_cond, expert_batch_dist_params = expert_batch
    if args.method.type == "sqil":
        # convert policy reward to 0
        online_batch_reward = torch.zeros_like(online_batch_reward)
        # convert expert reward to 1
        expert_batch_reward = torch.ones_like(expert_batch_reward)

    def change_shape(online, expert):
        shape = online.shape
        expert = torch.reshape(expert, shape)
        return expert
    if args.cond_dim!=-2 and args.expert.demos!=1:
        expert_batch_state = change_shape(online_batch_state, expert_batch_state)
        expert_batch_next_state = change_shape(online_batch_next_state, expert_batch_next_state)
        expert_batch_action = change_shape(online_batch_action, expert_batch_action)
        expert_batch_reward = change_shape(online_batch_reward, expert_batch_reward)
        expert_batch_done = change_shape(online_batch_done, expert_batch_done)

    batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
    batch_next_state = torch.cat(
        [online_batch_next_state, expert_batch_next_state], dim=0)
    batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
    batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
    batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
    batch_cond = torch.cat([online_batch_cond, expert_batch_cond], dim=0)
    is_expert = torch.cat([torch.zeros_like(online_batch_reward, dtype=torch.bool),
                           torch.ones_like(expert_batch_reward, dtype=torch.bool)], dim=0)

    # return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert
    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, batch_cond, is_expert

def save_state(tensor, path, num_states=5):
    """Show stack framed of images consisting the state"""

    tensor = tensor[:num_states]
    B, C, H, W = tensor.shape
    images = tensor.reshape(-1, 1, H, W).cpu()
    save_image(images, path, nrow=num_states)
    # make_grid(images)


def average_dicts(dict1, dict2):
    return {key: 1/2 * (dict1.get(key, 0) + dict2.get(key, 0))
                     for key in set(dict1) | set(dict2)}

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss + self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False