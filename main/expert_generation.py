# Sample script for preparing the dataset
import random
from itertools import count
from collections import defaultdict
# import gymnasium as gym
import gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
# from tensorboardX import SummaryWriter

from agent import make_agent
from make_envs import make_env, is_atari
from dataset.memory import Memory
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    env = make_env(args)
    if args.eval.use_baselines: # use baselines to train the model
        # from baselines_zoo.baselines_expert import BaselinesExpert
        # agent = BaselinesExpert(args.env.name, folder='rl-trained-agents')
        # env = agent.env
        if args.agent.name=="sac":
            print("--> Using SAC to train the model")
            from stable_baselines3 import SAC
            agent = SAC("MlpPolicy", env=env,verbose=1)
        else:
            print("--> Using PPO to train the model")
            from stable_baselines3 import PPO
            agent = PPO("MlpPolicy", env=env,verbose=1)
        TRAIN_TIMESTEPS = args.expert_gen_steps
        TOTAL_EPOCHS = 10
        print(f"Env short name : {args.env.short_name}")
        print(f"Env name : {args.env.name}")
        print("--> Training model, please expect a long training time")
        three_level = True
        if three_level:
            for i in range(TOTAL_EPOCHS):
                agent = agent.learn(total_timesteps=TRAIN_TIMESTEPS, log_interval=10)
                model_save_path = f'/home/zichang/proj/IQ-Learn/iq_learn/trained_policies/{args.env.short_name}/{(i+1)*TRAIN_TIMESTEPS}.zip'
                agent.save(model_save_path)
            # agent.load_state_dict(torch.load("/home/zichang/proj/IQ-Learn/iq_learn/iq.para/policy.pth"))
            
        else:
            cumu = 0
            TRAIN_TIMESTEPS = 14000
            cumu += TRAIN_TIMESTEPS
            agent = agent.learn(total_timesteps=TRAIN_TIMESTEPS, log_interval=10)
            model_save_path = f'/home/zichang/proj/IQ-Learn/iq_learn/trained_policies/{args.env.short_name}/{cumu}.zip'
            agent.save(model_save_path)
            TRAIN_TIMESTEPS = 4000
            cumu += TRAIN_TIMESTEPS
            agent = agent.learn(total_timesteps=TRAIN_TIMESTEPS, log_interval=10)
            model_save_path = f'/home/zichang/proj/IQ-Learn/iq_learn/trained_policies/{args.env.short_name}/{cumu}.zip'
            agent.save(model_save_path)
            TRAIN_TIMESTEPS = 2000
            cumu += TRAIN_TIMESTEPS
            agent = agent.learn(total_timesteps=TRAIN_TIMESTEPS, log_interval=10)
            model_save_path = f'/home/zichang/proj/IQ-Learn/iq_learn/trained_policies/{args.env.short_name}/{cumu}.zip'
            agent.save(model_save_path)
        exit("Training done, please rerun the script with use_baselines=False to generate expert trajectories")
    else: # load model directly
        if args.eval.policy:
            expert_file = f'{args.eval.policy}'
        print(f'Loading expert from: {expert_file}')
        expert_file = hydra.utils.to_absolute_path(expert_file)
        # path = "/home/zichang/proj/IQ-Learn/iq_learn/trained_policies/cartpole_seals_30000.zip"
        if args.agent.name=="sac":
            print("--> Using SAC to load the model")
            from stable_baselines3 import SAC
            # agent = SAC("MlpPolicy", env=env,verbose=1)
            agent = SAC.load(expert_file, env=env)
        else:
            print("--> Using PPO to load the model")
            from stable_baselines3 import PPO
            agent = PPO("MlpPolicy", env=env,verbose=1)
            agent = PPO.load(expert_file, env=env)
        # agent = make_agent(env, args)
        # agent.load("/home/zichang/proj/IQ-Learn/iq_learn/iq.para/actor.optimizer.pth",
        #        "/home/zichang/proj/IQ-Learn/iq_learn/iq.para/critic.optimizer.pth")

    # expert_file = f'{args.method.type}.para'
    # if args.eval.policy:
    #     expert_file = f'{args.eval.policy}'
    # print(f'Loading expert from: {expert_file}')

    # agent.load(hydra.utils.to_absolute_path(expert_file), f'_{args.env.name}')
    
    episode_reward = 0

    REPLAY_MEMORY = None
    REWARD_THRESHOLD = args.eval.threshold
    if args.eval.upper:
        REWARD_UPPER = args.eval.upper
    else:
        REWARD_UPPER = None
    MAX_EPS = args.expert.demos
    EPS_STEPS = int(args.env.eps_steps)

    memory_replay = Memory(REPLAY_MEMORY)
    total_steps = 0
    saved_eps = 0
    expert_trajs = defaultdict(list)
    expert_lengths = []
    expert_rewards = []

    for epoch in count():
        if saved_eps >= MAX_EPS:
            break
        if epoch > 300*MAX_EPS:
            print('Max episodes reached')
            break
        vec_env = agent.get_env()
        state = env.reset()
        
        # state = env.reset()
        # state = state[0]
        episode_reward = 0
        traj = []

        episode_infos = None
        for time_steps in range(EPS_STEPS):
            action, _states = agent.predict(state)
            # if action.min() < -1.0 or action.max() > 1.0:
            #     print(f"Action out of bound: {action.min()}, {action.max()}") 
            # action, _states = agent.predict(state, deterministic=True)
            # action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            if is_atari(args.env.name) and isinstance(action, np.ndarray):
                action = action.item()

            traj.append((state, next_state, action, reward, done))
            # print(reward)
            episode_reward += reward
            if memory_replay.size() == REPLAY_MEMORY:
                print('expert replay saved...')
                memory_replay.save(f'experts/{args.env.short_name}_{args.expert.demos}_{int(episode_reward)}')
                exit()

            state = next_state

            if is_atari(args.env.name):
                if 'ale.lives' in info:  # true for breakout, false for pong
                    done = info['ale.lives'] == 0
                episode_infos = info.get("episode")
                if episode_infos is not None:
                    episode_reward = episode_infos['r']
                    print("Atari Episode Length", episode_infos["l"])

            if done:
                # use_success = 'is_success' in info.keys()
                # score = info.get('is_success')
                break

        
        # if (not REWARD_THRESHOLD or episode_reward >= REWARD_THRESHOLD) and (not use_success or score >= 1.):
        if args.agent=="sac" and episode_reward.ndim == 1:
            episode_reward = episode_reward[0]
        if (not REWARD_THRESHOLD or episode_reward >= REWARD_THRESHOLD) and (not REWARD_UPPER or episode_reward <= REWARD_UPPER):
            saved_eps += 1
            states, next_states, actions, rewards, dones = zip(*traj)
            length = len(traj)

            pad = True # True for padding, False for no padding
            if pad == True:
                # pad them
                target_len = 1000
                states = padded(states, target_len)
                next_states = padded(next_states, target_len)
                actions = padded(actions, target_len)
                rewards = padded(rewards, target_len)
                dones = padded(dones, target_len)
                length = target_len

            expert_trajs["states"].append(states)
            expert_trajs["next_states"].append(next_states)
            expert_trajs["actions"].append(actions)
            expert_trajs["rewards"].append(rewards)
            expert_trajs["dones"].append(dones)
            expert_trajs["lengths"].append(length)
            expert_lengths.append(length)
            expert_rewards.append(episode_reward)
            print('Ep {}\tSaving Episode reward: {:.2f}\t'.format(epoch, episode_reward))
        else:
            print('Ep {}\tSkipped episode with reward: {:.2f}\t'.format(epoch, episode_reward))
        

    # for k, v in expert_trajs.items():
    #     expert_trajs[k] = np.array(v)
    if len(expert_trajs["lengths"]) > 0:
        get_data_stats(expert_trajs, np.array(expert_rewards), np.array(expert_lengths))
        print('Final size of Replay Buffer: {}'.format(sum(expert_trajs["lengths"])))
        mean_reward = int(np.mean(expert_rewards))
        save_path = f'experts/{args.env.short_name}/{args.env.name.replace("/","")}_{args.expert.demos}_{mean_reward}r.pkl'
        with open(hydra.utils.to_absolute_path(save_path), 'wb') as f:
            pickle.dump(expert_trajs, f)
        print(f'Expert traj saved in {save_path}')
    exit()


def get_data_stats(d, rewards, lengths):
    print("rewards: {:.2f} +/- {:.2f}".format(rewards.mean(), rewards.std()))
    print("len: {:.2f} +/- {:.2f}".format(lengths.mean(), lengths.std()))


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


if __name__ == '__main__':
    main()
