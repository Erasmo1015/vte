# Collect 100 trajs for each ability level. Designed for kitchen environment.
import argparse
import numpy as np
import pickle
from typing import Any, Dict, IO, List, Tuple
import torch
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
def open_datasets(paths):
    trajs_list = []
    for expert_location in paths:
        with open(expert_location, 'rb') as f:
            trajs = read_file(expert_location, f)
        trajs_list.append(trajs)
    return trajs_list

def padded(a, target_length, axis=0):
    """Add padding at end of to make array fixed size"""
    x = np.array(a)
    pad_size = target_length - x.shape[axis]
    axis_nb = len(x.shape)

    if pad_size < 0:
        return a
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    return np.pad(a, pad_width=npad, mode="constant", constant_values=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect datasets from given paths.")
    parser.add_argument('paths', type=str, help='List of paths to dataset files')
    parser.add_argument('num_trajs', type=int, help='Number of trajs for each ability level')
    parser.add_argument('save_path', type=str, default='experts/kitchen/kitchen_*.pkl')
    args = parser.parse_args()
    paths = args.paths.split(',')
    trajs_list = open_datasets(paths)
    max_reward = [max(i) for i in trajs_list[0]["rewards"]]
    print(f"Max reward max mean min: {max(max_reward)} {np.mean(max_reward)} {min(max_reward)}")
    # calculate the number of different values in max_reward
    unique_values = np.unique(max_reward)
    print(f"Number of unique values in max_reward: {len(unique_values)}")
    # count the number of occurences of each unique value
    occurences = {value: max_reward.count(value) for value in unique_values}
    print(f"Occurences of each unique value in max_reward: {occurences}")
    # collect 100 trajs for each ability level. e.g. ability level 0: max_reward=1, ability level 1: max_reward=2, etc.
    num_ability_levels = 3
    ability_levels = {i: {} for i in range(1, 1+num_ability_levels)}
    for ability_level in ability_levels.keys():
        collected_ids = [index for index, item in enumerate(trajs_list[0]["rewards"]) if max(item) == ability_level]
        
        for key in trajs_list[0].keys():
            ability_levels[ability_level][key] = []
            for i in collected_ids[:args.num_trajs]: 
                traj_content = trajs_list[0][key][i]
                if key not in "lengths" and len(traj_content) < 1000:
                        traj_content = padded(traj_content, 1000)
                        print(f"Padding {key} of length {len(traj_content)}")
                elif key in "lengths":
                    traj_content = 1000
                ability_levels[ability_level][key].append(traj_content)

        if len(collected_ids) < args.num_trajs:
            print(f"Warning: ability level {ability_level} has only {len(collected_ids)} trajs")
            collected_ids = [index for index, item in enumerate(trajs_list[0]["rewards"]) if max(item) == ability_level]
            for key in trajs_list[1].keys():
                for i in collected_ids[:args.num_trajs-len(ability_levels[ability_level]["rewards"])]:
                    traj_content = trajs_list[1][key][i]
                    if key not in "lengths" and len(traj_content) < 1000:
                        traj_content = padded(traj_content, 1000)
                        print(f"Padding {key} of length {len(traj_content)}")
                    elif key in "lengths":
                        traj_content = 1000
                    ability_levels[ability_level][key].append(traj_content)
        assert len(ability_levels[ability_level]["rewards"]) == args.num_trajs, f"Collected {len(ability_levels[ability_level]['rewards'])} trajs for ability level {ability_level}"
        print(f"Collected {args.num_trajs} trajs for ability level {ability_level}")
        
    # save each ability level to a file
    for ability_level in ability_levels.keys():
        save_file = args.save_path.replace('*', f"{args.num_trajs}_{ability_level}R")
        with open(save_file, 'wb') as f:
            pickle.dump(ability_levels[ability_level], f)
        print(f"Saved the data to {save_file}")
    
    # save the whole dataset to a file
    all_data = {}
    for key in trajs_list[0].keys():
        all_data[key] = []
        for ability_level in ability_levels.keys():
            all_data[key].extend(ability_levels[ability_level][key])
    assert len(all_data["rewards"]) == args.num_trajs * num_ability_levels, f"Collected {len(all_data['rewards'])} trajs in total"
    save_file = args.save_path.replace('*', f"{args.num_trajs * num_ability_levels}")
    with open(save_file, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"Saved the data to {save_file}")
