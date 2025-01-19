import os
import torch
from gcpc.model.policynet import PolicyNet  # Import PolicyNet class
import hydra
from omegaconf import DictConfig
import gym
from gcpc.model.utils import get_goal, suppress_output
import pytorch_lightning as pl
import pickle
from gcpc.data.parse_d4rl import parse_pickle_datasets, DATASET_DIR
import numpy as np
from .data.parse_d4rl import check_env_name, get_goal_type
from tqdm import tqdm
import pandas as pd
def evaluate_single_traj(gcpc, full_obs_seq, device, cfg, target_return, eval_eps=20):
    """
    Evaluates a single trajectory using the given policynet.

    Args:
        policynet: The trained policy network (PolicyLM).
        device: Torch device (cpu or cuda).
        cfg: Configuration containing parameters like environment, observation size, etc.

    Returns:
        Average episode reward.
    """

    # Set policynet in evaluation mode
    gcpc.eval()
    gcpc.model.to(device) # explicitly move trajnet model to device   
    gcpc.ctx_size = cfg.exp.ctx_size
    # Initialize environment
    env = gym.make(cfg.env_name)
    env.seed(cfg.seed)
    goal_type = get_goal_type(cfg.env_name)
    

    total_rewards = []
    total_steps = []
    full_obs_seq = torch.tensor(full_obs_seq, device=device, dtype=torch.float32)
    full_obs_seq = full_obs_seq.unsqueeze(0)
    for i in range(eval_eps):
        total_reward = 0 
        t = 0
        # Reset the environment and get initial observation
        ini_obs = env.reset()
        ini_obs = torch.tensor(ini_obs, device=device, dtype=torch.float32)[:gcpc.model.obs_dim]
        
        # Create a placeholder goal tensor with zeros (adjust size as needed)
        goal = torch.zeros(gcpc.model.goal_dim, device=device)
        goal = get_goal(env=env, goal_type=goal_type, goal_frac=1, obs=ini_obs.cpu().detach().numpy(), target_return=target_return)
        goal = torch.tensor(goal, device=device, dtype=torch.float)
        # make every value of goal as 0
        goal = torch.zeros_like(goal) # HACK GCPC without goal
        # goal = torch.ones_like(goal) 


        # Initialize evaluation
        obs, actions, goal = gcpc.init_eval(ini_obs, goal, gcpc.model.obs_dim, gcpc.model.action_dim, gcpc.model.goal_dim)

        done = False
        eval_horizon = env._max_episode_steps  # Assuming the environment has this attribute

        obs = obs.to(device)
        actions = actions.to(device)
        while not done and t < eval_horizon:
            # Get action from policy
            # action = gcpc.ar_step(t, obs, actions, goal)
            action = gcpc.ar_step_full(t, obs, full_obs_seq, actions, goal)  # HACK GCPC without goal

            # Step the environment
            next_obs, reward, done, _ = env.step(action.cpu().detach().numpy())
            total_reward += reward

            # Process the next observation
            next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)[:gcpc.model.obs_dim]
            # obs, actions = gcpc.ar_step_end(t, next_obs, action, obs, actions)
            obs, actions = gcpc.ar_step_end_full(t, next_obs, action, obs, actions) # HACK GCPC without goal

            t += 1
        total_rewards.append(total_reward)
        total_steps.append(t)
    # Return the total reward as the average episode reward
    return total_rewards, total_steps

def read_trajs(path):
    with open(path, 'rb') as f:
        trajs = pickle.load(f)
    
    return trajs

def avg_l2_norm(x, y):
    l2_norm = []
    for i in range(len(x)):
       l2_norm.append(np.sqrt((x[i]-y[i])**2))
    return np.mean(l2_norm), np.std(l2_norm)

@hydra.main(version_base=None, config_path='../config', config_name='train')
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define paths to the saved checkpoints
    gcpc = PolicyNet.load_from_checkpoint(checkpoint_path=cfg.pl_path, map_location=device)
    if "kitchen" not in cfg.env_name and "antmaze" not in cfg.env_name:
        dataset_path = os.path.join(DATASET_DIR, f'{cfg.env_name}.pkl')
        trajs = read_trajs(dataset_path)
        full_obs_seq = trajs['states']
        full_rew_seq =trajs['rewards']
    else:   
        dataset_path = os.path.join(DATASET_DIR, f'{cfg.env_name}.pkl')
        trajs = read_trajs(dataset_path)
        # full_obs_seq = trajs['observations']
        # full_rew_seq = trajs['rewards']
        full_obs_seq = []
        full_rew_seq = []
    results = []
    step_list = []
    expert_rew = []
    # indexes = [range(0,10)].extend(range(100,110)).extend(range(200,210))
    indexes = range(len(full_obs_seq))
    print(f"Total number of trajectories: {len(full_obs_seq)}")
    for index in tqdm(indexes):
        if "kitchen" not in cfg.env_name and "antmaze" not in cfg.env_name:
            target_return = np.sum(full_rew_seq[index])
            obs_seq = full_obs_seq[index]
        else:
            target_return = 0
            obs_seq = []
            target_return = None
        avg_return_per_episode, steps = evaluate_single_traj(gcpc, obs_seq, device, cfg, target_return)
        # print(f'Episode reward: {avg_return_per_episode} , Steps: {steps}')
        results.append(avg_return_per_episode)
        step_list.append(steps)
        expert_rew.append(target_return)
    # for i in range(len(results)):
    #     print(f"Traj {indexes[i]}: Reward: {results[i]}, Steps: {step_list[i]}, Expert Reward: {expert_rew[i]}")

    save_dict = {'results': results, 'expert_rew': expert_rew}
    CSV_DIR = "results/csv/"
    df = pd.DataFrame(save_dict)
    csv_filename = os.path.join(CSV_DIR, f'{cfg.env_name}_{cfg.seed}.csv') 
    df.to_csv(csv_filename, index=False)
    print(f"Rewards has been saved to '{csv_filename}'.")

    # mean, std = avg_l2_norm(results,expert_rew)
    # print(f"L2 Norm Mean: {mean}, Std: {std}")

if __name__ == '__main__':
    main()
