from typing import Any, Dict, IO, List, Tuple

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import os


class ExpertDataset(Dataset):
    """Dataset for expert trajectories.

    Assumes expert dataset is a dict with keys {states, actions, rewards, lengths} with values containing a list of
    expert attributes of given shapes below. Each trajectory can be of different length.

    Expert rewards are not required but can be useful for evaluation.

        shapes:
            expert["states"]  =  [num_experts, traj_length, state_space]
            expert["actions"] =  [num_experts, traj_length, action_space]
            expert["rewards"] =  [num_experts, traj_length]
            expert["lengths"] =  [num_experts]
    """

    def __init__(self,
                 expert_location: str,
                 num_trajectories: int = 4,
                 subsample_frequency: int = 20,
                 seed: int = 0,
                 cond_dim: int = 10,
                 cond_type: str = "random",
                 conds: dict = None):
        """Subsamples an expert dataset from saved expert trajectories.

        Args:
            expert_location:          Location of saved expert trajectories.
            num_trajectories:         Number of expert trajectories to sample (randomized).
            subsample_frequency:      Subsamples each trajectory at specified frequency of steps.
            deterministic:            If true, sample determinstic expert trajectories.
        """
        self.cond_dim = cond_dim
        self.cond_type = cond_type
        all_trajectories, perm = load_trajectories(expert_location, num_trajectories, seed)
        self.trajectories = {}

        # Randomize start index of each trajectory for subsampling
        # start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

        # Subsample expert trajectories with every `subsample_frequency` step.
        for k, v in all_trajectories.items():
            data = v

            if k != "lengths":
                samples = []
                num_trajectories = min(num_trajectories, len(data))
                for i in range(num_trajectories):
                    samples.append(data[i][0::subsample_frequency])
                self.trajectories[k] = samples
            else:
                # Adjust the length of trajectory after subsampling
                self.trajectories[k] = np.array(data) // subsample_frequency

        self.i2traj_idx = {}
        self.length = self.trajectories["lengths"].sum().item()

        del all_trajectories  # Not needed anymore
        traj_idx = 0
        i = 0

        # Convert flattened index i to trajectory indx and offset within trajectory
        self.get_idx = []

        for _j in range(self.length):
            while self.trajectories["lengths"][traj_idx].item() <= i:
                i -= self.trajectories["lengths"][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))
            i += 1
        # print("conds length: ", len(self.conds))
        # print("trajectories length: ", len(self.trajectories["states"]))
        # apply permutation to cond
        # print("perm:",perm)
        if cond_type!="none":
            if type(conds["emb"][0])==torch.Tensor:
                self.conds = [conds["emb"][i].detach().cpu().numpy() for i in perm]
            else:
                self.conds = [conds["emb"][i] for i in perm]
            self.conds = self.conds[:num_trajectories]
            self.true_traj_idx_list = perm[:num_trajectories]
        else:
            self.conds = conds["emb"][:num_trajectories]
            self.true_traj_idx_list = perm[:num_trajectories]
        # print("permuted condss:",self.conds)
        assert len(self.conds)==len(self.trajectories["states"])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]
        # if self.cond_type=="fixed":
        #     traj_idx = 0
        if traj_idx<=len(self.conds):
            cond = self.conds[traj_idx]
            true_traj_idx = self.true_traj_idx_list[traj_idx]
        else:
            raise ValueError(f"Trajectory index {traj_idx} out of range")
        states = self.trajectories["states"][traj_idx][i]
        next_states = self.trajectories["next_states"][traj_idx][i]

        # Rescale states and next_states to [0, 1] if are images
        if isinstance(states, np.ndarray) and states.ndim == 3:
            states = np.array(states) / 255.0
        if isinstance(states, np.ndarray) and next_states.ndim == 3:
            next_states = np.array(next_states) / 255.0

        # cond = [-1]*self.cond_dim
        # if len(cond)<self.cond_dim:
        #     raise ValueError(f"cond_dim {self.cond_dim}out of range, maximum cond length is {len(cond)}")
        if self.cond_dim > 0 and (self.cond_type=="random" or self.cond_type=="debug"):
            cond = cond[:self.cond_dim]
        elif self.cond_type=="none" or self.cond_type=="dummy":
            cond = [-1]*self.cond_dim
        return (states,
                next_states,
                self.trajectories["actions"][traj_idx][i],
                self.trajectories["rewards"][traj_idx][i],
                self.trajectories["dones"][traj_idx][i], cond, true_traj_idx)


def load_trajectories(expert_location: str,
                      num_trajectories: int = 10,
                      seed: int = 0) -> Dict[str, Any]:
    """Load expert trajectories

    Args:
        expert_location:          Location of saved expert trajectories.
        num_trajectories:         Number of expert trajectories to sample (randomized).
        deterministic:            If true, random behavior is switched off.

    Returns:
        Dict containing keys {"states", "lengths"} and optionally {"actions", "rewards"} with values
        containing corresponding expert data attributes.
    """
    if os.path.isfile(expert_location):
        # Load data from single file.
        with open(expert_location, 'rb') as f:
            trajs = read_file(expert_location, f)

        rng = np.random.RandomState(seed)
        # Sample random `num_trajectories` experts.
        perm = np.arange(len(trajs["states"]))
        perm = rng.permutation(perm) # FIXME: disable random permutation for now. Can be enabled if use the traj encoder. 
        num_trajectories = min(num_trajectories, len(perm))
        idx = perm[:num_trajectories]
        for k, v in trajs.items():
            # if not torch.is_tensor(v):
            #     v = np.array(v)  # convert to numpy array
            trajs[k] = [v[i] for i in idx]
    else:
        raise ValueError(f"{expert_location} is not a valid path")
    return trajs, perm


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
