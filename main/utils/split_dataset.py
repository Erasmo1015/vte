# Convert GCPC dataset to IQ learn format
import numpy as np
import pickle
import os
import argparse
def open_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories

def save_split_dataset(split_data, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"Saved split dataset to {output_file}")

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

def preprocess_gcpc_to_love_format(path: str, episode_num: int):
    '''
    Read GCPC dataset and return states and actions in LOVE-compatible format.
    '''
    with open(path, 'rb') as f:
        episodes = pickle.load(f)

    n_episode = len(episodes)
    save_n_episode = min(n_episode, episode_num)
    print(f'Loading dataset from {path}: {n_episode} episodes')

    # Extract the key dimensions for padding (optional)
    episode_lengths = [e['rewards'].shape[0] for e in episodes]
    
    # Initialize lists to store states and actions
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    goals = []
    lengths = []

    pad = True # NOTE True for padding, False for no padding
    # Loop through the episodes and collect states and actions
    for e in episodes[:save_n_episode]:
        # make next_observations a list of observations that have all indexes shifted +1
        next_observations = np.concatenate([e['observations'][1:], e['observations'][-1:]], axis=0)
        if pad is True:
            # pad them
            target_len = max(1000, e['observations'].shape[0])
            observation = padded(e['observations'], target_len)
            action = padded(e['actions'], target_len)
            next_observation = padded(next_observations, target_len)
            reward = padded(e['rewards'], target_len)
            done = padded(e['terminals'], target_len)
            info = padded(e['infos'], target_len) # infos for kitchen and infos/goal for antmaze
            length = target_len
        else:
            # no padded
            observation = e['observations']
            action = e['actions']
            next_observation = next_observations
            reward = e['rewards']
            done = e['terminals']
            info = e['infos']
            length = e['observations'].shape[0]
        # for states, actions, next_states, rewards, dones, infos/goal in e:
        states.append(np.array(observation, dtype=np.float32))
        actions.append(np.array(action, dtype=np.float32))
        next_states.append(np.array(next_observation, dtype=np.float32))
        rewards.append(np.array(reward, dtype=np.float32))
        dones.append(np.array(done, dtype=np.float32))
        goals.append(np.array(info, dtype=np.float32))
        lengths.append(length)


        
        
    new_dataset = {
        'states': states,
        'actions': actions,
        'next_states': next_states,
        'rewards': rewards,
        'dones': dones,
        'infos': goals,
        'lengths': lengths
    }
    # Return states and actions in the format expected by LOVE
    return new_dataset

def main():
    # make expert_location an argument
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("file", type=str, help="name of the file")
    # arg_parser.add_argument("save_dir", type=str, help="name of the save directory")
    args = arg_parser.parse_args()
    # Example usage
    expert_file = args.file
    # Load dataset
    dataset_paths = expert_file.split(",")
    goal_dict = {}
    trajectories = open_dataset(dataset_paths[0])
    split_state = [len(traj['observations']) for traj in trajectories]
    print(f" max min and avg length of state: {max(split_state)} {min(split_state)} {np.mean(split_state)}")
    print(f"Number of trajectories: {len(trajectories)}")
    keys = trajectories[0].keys()
    print(f"Keys in the dataset: {keys}")
    print(f"Number of trajectories: {len(trajectories)}")
    # Prepare lists to store the split data

    # The current path with 'gcpc' in the argument
    original_path = args.file

    # Remove 'gcpc/' from the original path
    modified_path = original_path.replace('gcpc/', '')

    # Now you have the new save path without 'gcpc'
    print(f"Modified path: {modified_path}")

    # Example: Save two files by modifying the number dynamically
    file_name_1 = modified_path.replace('v0', str(len(trajectories)))  # Replace v2 with 1
    file_name_2 = modified_path.replace('v0', '30')  # Replace v2 with 2


    for idx, dataset_path in enumerate(dataset_paths):
        new_dataset = preprocess_gcpc_to_love_format(dataset_path, 5000)
    # save the new dataset to save_dir
    save_split_dataset(new_dataset, file_name_1)


    for idx, dataset_path in enumerate(dataset_paths):
        new_dataset = preprocess_gcpc_to_love_format(dataset_path, 30)    
    # save the new dataset to save_dir
    save_split_dataset(new_dataset, file_name_2)

if __name__ == "__main__":
    
    main()
