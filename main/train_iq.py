"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""
import datetime
import os
import random
import time
import types
from collections import deque
from itertools import count

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from agent import make_agent
from dataset.memory import Memory
from iq import iq_loss
from make_envs import make_env
from utils.logger import Logger
from utils.utils import (average_dicts, eval_mode, evaluate,
                         get_concat_samples, hard_update, soft_update)
from wrappers.atari_wrapper import LazyFrames
import types
from dataset.memory import Memory
from make_envs import make_env
from utils.logger import Logger
from utils.utils import (average_dicts, eval_mode, evaluate,
                         get_concat_samples, hard_update, soft_update)
from utils.utils import EarlyStopper
from wrappers.atari_wrapper import LazyFrames
from typing import IO, Any, Dict
import pickle
from bc import BehaviorCloningLossCalculator
# from encoder.utils import *
from encoder.utils import cheetah_full_loader
from torch.optim import Adam
import sys
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(2)

def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    cfg.env_name = cfg.env_name
    cfg.agent_name = cfg.agent_name
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    if args.wandb:
        if args.cond_type!="none":
            if args.experimental != "none":
                exp_name = args.env.cond
                wandb.init(
                    # mode="offline",
                    project="hil_iq", 
                    sync_tensorboard=True, 
                    reinit=True, 
                    config=args, 
                    name=f"{args.env.short_name} experimental {args.experimental}"
                )
            elif args.method.bc_init:
                exp_name = args.env.cond
                wandb.init(
                    # mode="offline",
                    project="hil_iq", 
                    sync_tensorboard=True, 
                    reinit=True, 
                    config=args, 
                    name=f"{args.env.short_name} bc_init{args.method.bc_init} level{args.num_levels}"
                )
            else:
                level = "" if args.num_levels < 3 else f"level{args.num_levels} "
                exp_name = f"{args.env.short_name} bc_init{args.method.bc_init} {level}actor{args.agent.actor_lr} critic{args.agent.critic_lr}"
                wandb.init(
                    # mode="offline",
                    project="hil_iq", 
                    sync_tensorboard=True, 
                    reinit=True, 
                    config=args, 
                    name=exp_name
                )
        else: 
            exp_name = args.env.demo
            wandb.init(
                project="hil_iq", 
                sync_tensorboard=True, 
                reinit=True, 
                config=args, 
                name=f"{args.env.name} iq-learn only"
            )
    
    if (not args.method.bc_init):
        print(f'[Critic]: use {args.additional_loss}*{args.cql_coef} ({args.num_random} randoms) loss')

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env
    env = make_env(args)
    eval_env = make_env(args)

    # Seed envs
    env.seed(args.seed) # TODO: uncomment this to enable seed on gym
    eval_env.seed(args.seed + 10)

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)
    INITIAL_STATES = 128  # Num initial states to use to calculate value of initial state distribution s_0

    agent = make_agent(env, args)
    if args.pretrain:
        pretrain_paths = args.pretrain
        if len(pretrain_paths) == 2:
            actor_path, critic_path = pretrain_paths
            actor_path = hydra.utils.to_absolute_path(actor_path)
            critic_path = hydra.utils.to_absolute_path(critic_path)

            if os.path.isfile(actor_path) and os.path.isfile(critic_path):
                print("=> loading pretrain '{}' and '{}'".format(actor_path, critic_path))
                agent.load(actor_path, critic_path)
            else:
                print("[Attention]: One or both checkpoints not found: {} and {}".format(actor_path, critic_path))
        else:
            # print("[Error]: Invalid number of paths provided. Expected two paths in a list.")
            pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
            if os.path.isfile(pretrain_path):
                print("=> loading pretrain '{}'".format(args.pretrain))
                agent.load(pretrain_path)
            else:
                print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    # Load expert data
    conds = None
    cond_location = hydra.utils.to_absolute_path(f'cond/{args.env.cond}')
    if os.path.isfile(cond_location):
        # Load data from single file.
        with open(cond_location, 'rb') as f:
            conds = read_file(cond_location, f)
    else:
        raise ValueError(f"Condition file {cond_location} not found")
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42,
                              cond_dim=args.cond_dim,
                              cond_type=args.cond_type,
                              conds=conds)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)

    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir,
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=True,
                    agent=args.agent.name)

    steps = 0

    # track mean reward and scores
    scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    best_eval_returns = -np.inf

    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    # Encoder Init
    # print work dir
    # change work dir to encoder dir
    # os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    # if sac
    if args.agent.name == "sac":
        agent.loss_calculator = BehaviorCloningLossCalculator(
                ent_weight=1e-3,  # args.method.bc_ent_weight,
                l2_weight=0.0,  # args.method.bc_l2_weight,
                kld_weight=args.method.kld_alpha,  # args.method.bc_kld_weight,
                agent_name=args.agent.name
            )
    else: 
        # TODO: add bc loss calculator for softq
        agent.loss_calculator = BehaviorCloningLossCalculator(
                ent_weight=1e-3,  # args.method.bc_ent_weight,
                l2_weight=0.0,  # args.method.bc_l2_weight,
                kld_weight=args.method.kld_alpha,  # args.method.bc_kld_weight,
                agent_name=args.agent.name
            )
    agent.bc_alpha = args.method.bc_alpha
    if args.method.bc_init:
        # sys.path.append('/common/home/users/z/zichang.ge.2023/proj/IQ-Learn/iq_learn/encoder') # NOTE for origami server
        sys.path.append('/home/zichang/proj/IQ-Learn/iq_learn/encoder')
        print("Current working directory: ", os.getcwd())
        exp_dir = args.exp_dir
        checkpoint = exp_dir + args.encoder
        encoder = torch.load(checkpoint)
        encoder.train()
        encoder.instantiate_prob_encoder(dist_size=args.cond_dim)
        encoder.to(device)

        # check if seq_dim is 1000:
        with open(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'), 'rb') as f:
            trajs = read_file(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'), f)
            seq_dim = len(trajs['states'][0])
            print(f"Traj sequence dim: {seq_dim}")
        
        last_layers_to_unfreeze = ['z_logit_feat', 'm_feat', 'transformer', 'compact_last', 'mu_layer','logvar_layer']

        # Freeze all parameters first
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze specific last layers by their names
        for name, param in encoder.named_parameters():
            if any(last_layer_name in name for last_layer_name in last_layers_to_unfreeze):
                param.requires_grad = True

        # Print all layers and sub-layers
        # print("\nModel layers using model.modules():")
        # for layer in encoder.modules():
        #     print(layer)


        encoder_optimizer = Adam(params=encoder.parameters(), lr=3e-05, amsgrad=True)
        print(f"Encoder Loaded: {checkpoint}, optimizer ready")
        # bc loss function
        # if args.method.enable_bc_actor_update:
        # agent.loss_calculator = BehaviorCloningLossCalculator(
        #     ent_weight=1e-3,  # args.method.bc_ent_weight,
        #     l2_weight=0.0,  # args.method.bc_l2_weight,
        #     kld_weight=args.method.kld_alpha,  # args.method.bc_kld_weight,
        # )
        # agent.bc_alpha = args.method.bc_alpha
        # encoder is model.ckpt, remove ".ckpt"
        encoder_name = args.encoder.split(".")[0]
        # find parent folder of checkpoint and give to exp_id
        exp_id = os.path.basename(os.path.dirname(args.exp_dir))
        # unique_temp_cond_file = f"cond/{args.env.short_name}/{exp_id}/temp_{args.env.name}_{encoder_name}.pkl"
        # # unique_temp_cond_file= "cond/temp_cond.pkl"
        # print(f"-> Unique temp cond file: {unique_temp_cond_file}")
        
    # BC initialization
    if args.method.bc_init:
        save_encoder_dir = exp_dir
        expert_file = hydra.utils.to_absolute_path(f'experts/{args.env.demo}')
        agent.bc_update = types.MethodType(bc_update, agent)
        logit_m = conds["logit_m"]
        logit_array_0, m_array_0 = logit_m[0]
        # if logit_array_0, m_array_0 is not tensor, put them to tensor device
        # if not torch.is_tensor(logit_array_0):
        #     logit_array_0 = torch.tensor(logit_array_0, device=device)
        #     m_array_0 = torch.tensor(m_array_0, device=device)
        first_mu_0 = get_mu_logvar(logit_array_0, m_array_0, encoder, device)[1].detach().cpu().numpy()
        early_stopper = EarlyStopper(patience=3, min_delta=-0.2)
        for learn_steps_bc in count():
            # print(f"BC step: {learn_steps_bc}")
            expert_batch = expert_memory_replay.get_samples(
                agent.batch_size, agent.device
            )
            expert_obs, _, expert_action, __, ___, expert_cond_detached, true_traj_idx = expert_batch
            emb_list = get_new_cond(
                encoder, 
                logit_m, 
                device,
                true_traj_idx)
            unique = []
            dist_params_list = []
            for i in true_traj_idx:
                if i not in unique:
                    unique.append(i)
                    dist_params_list.append(emb_list["dist_params"][i])
            mu = [dist_params[0] for dist_params in dist_params_list]     
            log_var = [dist_params[1] for dist_params in dist_params_list]
            mu = torch.stack(mu, dim=0)
            log_var = torch.stack(log_var, dim=0)
           
            expert_cond_list = emb_list["emb"]
            expert_cond = torch.stack(expert_cond_list, dim=0)
            
            # epsilon = 1e-6  # Small value to avoid numerical issues
            # if args.agent.name=="sac" and (expert_action.min() <= -1.0 or expert_action.max() >= 1.0):
            #     print(f"Action out of bound: {expert_action.min()}, {expert_action.max()}") 
            #     print("Force normalizing expert action to [-1, 1]")
            #     expert_action = torch.clamp(expert_action, min=-1.0 + epsilon, max=1.0 - epsilon)
            #     print(expert_action.min(), expert_action.max())

            losses = agent.bc_update(
                expert_obs,
                expert_action,
                expert_cond,
                logger,
                learn_steps_bc,
                args.cond_type,
                mu,
                log_var,
                encoder_optimizer,
                encoder
            )

            # for name, param in encoder.named_parameters():
            #     if any(last_layer_name in name for last_layer_name in last_layers_to_unfreeze):
            #         print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")

            # log losses
            
            if learn_steps_bc % 10 == 0:  # args.log_interval == 0:
                # test: log the l2 norm between the latest latent mean (traj_id=0) and the first latent mean every 10 steps
                _, mu_0, __ = get_mu_logvar(logit_array_0, m_array_0, encoder, device)
                mu_0 = mu_0.detach().cpu().numpy()
                l2_norm = np.linalg.norm(mu_0 - first_mu_0)
                
                # print(f"Step: {learn_steps_bc}, L2 Norm: {l2_norm.item()}")
                losses["mean_l2_norm"] = l2_norm.item()
                for key, loss in losses.items():
                    writer.add_scalar(key, loss, global_step=learn_steps_bc)
                
                logger.dump(learn_steps_bc)
                # print (f"Step: {learn_steps_bc}, L2 Norm: {l2_norm.item()}")
                if early_stopper.early_stop(-l2_norm.item()):  
                    # unique_encoder_file = f"prob-encoder_dim{args.cond_dim}_kld_alpha{args.method.kld_alpha}_betaB_step_{learn_steps_bc}.ckpt"
                    # save_dir = os.path.join(exp_dir, unique_encoder_file)
                    # torch.save(encoder, save_dir)
                    # print(f"Encoder saved at {save_dir}") 
                    # pass # NOTE remove to allow early stopping
                    print("Early stopping at step: ", learn_steps_bc)  
                    exit_save(encoder, learn_steps_bc, save_encoder_dir, expert_file, device, seq_dim, args)       
                    break 

            # eval every n steps
            if learn_steps_bc % args.env.eval_interval == 0:
                eval_num = 0 
                for eval_index in range(eval_num):
                    # low ability level
                    eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=eval_index, experimental="none")
                    returns = np.mean(eval_returns)
                    logger.log(f'eval/episode_reward_low{eval_index}', returns, learn_steps)
                    # high ability level
                    high_index = eval_index + args.expert.demos//2
                    eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=high_index, experimental="none")
                    returns = np.mean(eval_returns)
                    logger.log(f'eval/episode_reward_high{high_index}', returns, learn_steps)
                # logger.log('eval/bc_episode_reward', returns, learn_steps_bc)
                logger.dump(learn_steps_bc, ty="eval")
                # refresh expert memory with new cond and new dist_params
                emb_list = update_expert_memory(
                    encoder, 
                    hydra.utils.to_absolute_path(f'experts/{args.env.demo}'), 
                    device,
                    seq_dim=seq_dim)
                logit_m = emb_list["logit_m"]
                expert_memory_replay.clear()
                expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42,
                              cond_dim=args.cond_dim,
                              cond_type=args.cond_type,
                              conds=emb_list)
                print(f'--> New expert memory size: {expert_memory_replay.size()}')
            # save the encoder every 500 steps
            if (learn_steps_bc % args.bc_save_interval == 0 or learn_steps_bc==10) and learn_steps_bc > 0:
                exit_save(encoder, learn_steps_bc, save_encoder_dir, expert_file, device, seq_dim, args)
            # test: calculate the l2 norm between the latest latent mean and the first latent mean every 10 steps
            if learn_steps_bc == args.bc_steps:
                learn_steps_bc += 1
                print("Finished BC!")
                break
        exit_save(encoder, learn_steps_bc, save_encoder_dir, expert_file, device, seq_dim, args)
        sys.path.append('/common/home/users/z/zichang.ge.2023/proj/IQ-Learn/iq_learn/')
        print("Current working directory: ", os.getcwd())
        return # TODO: remove this to enable IQ-learn after bc-init
    print("Start IQ-learn")
    experiment_time = time.time()
    for epoch in count(): # n of episodes
        state = env.reset()
        episode_reward = 0
        done = False
        cond = get_random_cond(args.cond_dim, args.cond_type, conds)
        start_time = time.time()
        for episode_step in range(EPISODE_STEPS): # n of steps
            if "kitchen" in args.env.name:
                state = state[:30] # HACK kitchen GCPC dataset
            if steps < args.num_seed_steps:
                # Seed replay buffer with random actions
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    if args.cond_type=="none":
                        action = agent.choose_action(state, sample=True)
                    else:
                        action = agent.choose_action((state, cond), sample=True)
            next_state, reward, done, info = env.step(action)
            if "kitchen" in args.env.name:
                next_state = next_state[:30] # HACK kitchen GCPC dataset
            episode_reward += reward
            steps += 1

            if learn_steps % args.env.eval_interval == 0:
                if args.cond_type=="debug":
                    num_levels = args.num_levels
                    eval_num = 1
                    for eval_index in range(eval_num):
                        # low ability level
                        for level in range(num_levels):
                            current_index = eval_index + level*args.expert.demos//num_levels
                            eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=current_index, experimental="none")
                            returns = np.mean(eval_returns)
                            logger.log(f'eval/episode_reward_{current_index}', returns, learn_steps)
                        # eval_returns, eval_timesteps = evaluate(agent, eval_env, hydra.utils.to_absolute_path(f'cond/{args.env.cond}'), num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=eval_index)
                        # returns = np.mean(eval_returns)
                        # logger.log(f'eval/episode_reward_low{eval_index}', returns, learn_steps)
                        # # high ability level
                        # high_index = eval_index + args.expert.demos//num_levels
                        # eval_returns, eval_timesteps = evaluate(agent, eval_env, hydra.utils.to_absolute_path(f'cond/{args.env.cond}'), num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=high_index)
                        # returns = np.mean(eval_returns)
                        # logger.log(f'eval/episode_reward_high{high_index}', returns, learn_steps)
                    if args.experimental != "none":
                        eval_index = 10
                        eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=eval_index, experimental=args.experimental)
                        returns = np.mean(eval_returns)
                        logger.log(f'eval/episode_reward_{args.experimental}{eval_index}', returns, learn_steps)
                else:
                    eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, experimental="none")
                    returns = np.mean(eval_returns)
                    logger.log('eval/episode_reward', returns, learn_steps)
                learn_steps += 1  # To prevent repeated eval at timestep 0
                logger.log('eval/episode', epoch, learn_steps)
                logger.dump(learn_steps, ty='eval')
                # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

                if returns > best_eval_returns:
                    # Store best eval returns
                    best_eval_returns = returns
                    if wandb.run:
                        wandb.run.summary["best_returns"] = best_eval_returns
                    save(agent, epoch, args, output_dir='results_best')

            # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            # if type(state) == np.ndarray:
                # online_memory_replay.add((state, next_state, action, reward, done_no_lim, cond))
            online_traj_idx = -1
            online_memory_replay.add((state, next_state, action, reward, done_no_lim, cond, online_traj_idx))

            if online_memory_replay.size() > INITIAL_MEMORY:
                # Start learning
                if begin_learn is False:
                    print('Learn begins!')
                    begin_learn = True

                learn_steps += 1
                if learn_steps == LEARN_STEPS:
                    # Save the last model
                    if args.save_last:
                        exp_dir = args.exp_dir
                        result_last_dir = os.path.join(exp_dir, "result_last")

                        if not os.path.exists(result_last_dir):
                            os.makedirs(result_last_dir)
                            print(f"Created directory {result_last_dir}")
                        else:
                            print(f"Directory {result_last_dir} already exists")
                        # name the loc with timestamp
                        ts_str = datetime.datetime.fromtimestamp(experiment_time).strftime("%Y-%m-%d_%H-%M-%S")
                        save_loc = os.path.join(result_last_dir, f"{ts_str}_last")

                        agent.save(save_loc)
                        print(f"Saved model at {save_loc}")    

                    print('Finished!')  
                    wandb.finish()
                    return
                if learn_steps % 100000 == 0:
                    # Save the last model
                    if args.save_last:
                        exp_dir = args.exp_dir
                        result_last_dir = os.path.join(exp_dir, "result_last")

                        if not os.path.exists(result_last_dir):
                            os.makedirs(result_last_dir)
                            print(f"Created directory {result_last_dir}")
                        else:
                            print(f"Directory {result_last_dir} already exists")
                        # name the loc with timestamp
                        ts_str = datetime.datetime.fromtimestamp(experiment_time).strftime("%Y-%m-%d_%H-%M-%S")
                        save_loc = os.path.join(result_last_dir, f"{ts_str}")

                        agent.save(save_loc)
                        print(f"Saved model at {save_loc}")    
                    

                ######
                # IQ-Learn Modification
                agent.iq_update = types.MethodType(iq_update, agent)
                agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
                losses = agent.iq_update(online_memory_replay,
                                         expert_memory_replay, logger, learn_steps, args.cond_type)
                ######

                if learn_steps % args.log_interval == 0:
                    for key, loss in losses.items():
                        writer.add_scalar(key, loss, global_step=learn_steps)

            if done:
                break
            state = next_state

        rewards_window.append(episode_reward)
        # logger.log('train/episode', epoch, learn_steps)
        # logger.log('train/episode_reward', episode_reward, learn_steps)
        # logger.log('train/duration', time.time() - start_time, learn_steps)
        # logger.dump(learn_steps, save=begin_learn)
        # print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(agent, epoch, args, output_dir='results')

        used_time = time.time() - start_time
        # Convert to hours, minutes, seconds
        hours, rem = divmod(used_time, 3600)
        minutes, seconds = divmod(rem, 60)
        # Print in a readable format
        # print(f"Epoch {epoch} Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
def exit_save(encoder, learn_steps_bc, save_loc, expert_file, device, seq_dim, args):
    unique_encoder_file = f"prob-encoder_dim{args.cond_dim}_kld_alpha{args.method.kld_alpha}_betaB_step_{learn_steps_bc}.ckpt"
    save_dir = os.path.join(save_loc, unique_encoder_file)
    torch.save(encoder, save_dir)
    print(f"Encoder saved at {save_dir}")

    new_conds = update_expert_memory(encoder, expert_file, device, seq_dim=seq_dim)
    save_cond_loc = f"cond/{args.env.short_name}/{args.env.short_name}_{args.encoder}_step{learn_steps_bc}.pkl"
    save_cond_loc = hydra.utils.to_absolute_path(save_cond_loc)
    with open(save_cond_loc, 'wb') as f:
        pickle.dump(new_conds, f)
    print(f"New cond file saved at: {save_cond_loc}")

    
def get_mu_logvar(logit_arrays, m_arrays, encoder, device):
    logit_arrays = np.array(logit_arrays)
    m_arrays = np.array(m_arrays)
    logit = torch.tensor(logit_arrays).to(device)
    m = torch.tensor(m_arrays).to(device)
    cond = encoder.get_dist(logit, m)
    mu, logvar = encoder.get_dist_params()
    return cond, mu, logvar

def get_new_cond(encoder, logit_m, device, traj_idx_list):
    new_emb_list = {"emb": [], "dist_params":[], "logit_m":[]}
    for i in traj_idx_list:
        logit_arrays, m_arrays = logit_m[i]
        cond, mu, logvar = get_mu_logvar(logit_arrays, m_arrays, encoder, device)
        new_emb_list["emb"].append(cond)
        new_emb_list["dist_params"].append((mu, logvar))
    return new_emb_list

def update_expert_memory(encoder, expert_file, device, seq_dim=1000):
    full_loader = cheetah_full_loader(1, expert_file, seq_dim=seq_dim)
    encoder.post_obs_state._output_normal = True
    encoder._output_normal = True
    seq_size = full_loader.dataset.seq_size
    init_size = 1
    emb_list = {"num_m":[],"emb": [], "level":[], "num_z":[], "z":[], "dist_params":[], "logit_m":[]}
    for index, (obs_list, action_list, level_list) in enumerate(full_loader):
        # obs_list 100 1000 17
        # action_list 100 1000 6
        # trai_level_list 100
        obs_list = obs_list.to(device)
        action_list = action_list.to(device)
        results = encoder(obs_list, action_list, seq_size, init_size)
        mean, std = encoder.get_dist_params()
        emb = reparameterize(mean, std)
        # emb = emb.detach().cpu().numpy()
        logit_arrays, m_arrays = encoder.get_logit_m()
        emb_list["num_m"].extend([len(i) for i in results[-4]])
        # emb_list["emb"].extend(results[-3])
        emb_list["emb"].extend(emb)
        emb_list["level"].extend(level_list)
        emb_list["num_z"].extend([len(i) for i in results[-2]])
        emb_list["z"].extend(results[-1])
        emb_list["dist_params"].append((mean, std))
        emb_list["logit_m"].append((logit_arrays, m_arrays))
    ## --> Normalize the emb using z score normalization
    # emb_list["emb"] = zscore(emb_list["emb"])
    # save numpy arrays for file
    emb_list["emb"] = [i.detach().cpu().numpy() for i in emb_list["emb"]]
    emb_list["dist_params"] = [(i[0].detach().cpu().numpy(), i[1].detach().cpu().numpy()) for i in emb_list["dist_params"]]
    return emb_list

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

# cond_type: 1 for real indexed, 0 for fixed index 0, -1 for [-1]*cond_dim
def get_random_cond(cond_dim, cond_type, conds):
    conds = conds["emb"]
    # select random index from conds length
    index = random.randint(0, len(conds)-1)
    if cond_type=="random" or cond_type=="debug" :
        cond = conds[index][:cond_dim]
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

def save(agent, epoch, args, output_dir='results'):
    if epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f'sqil_{args.env.name}'
        else:
            name = f'iq_{args.env.name}'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f'{output_dir}/{args.agent.name}_{name}')


# Minimal IQ-Learn objective
def iq_learn_update(self, policy_batch, expert_batch, logger, step):
    args = self.args
    # policy_obs, policy_next_obs, policy_action, policy_reward, policy_done, policy_cond = policy_batch
    # expert_obs, expert_next_obs, expert_action, expert_reward, expert_done, expert_cond = expert_batch
    (
        policy_obs,
        policy_next_obs,
        policy_action,
        policy_reward,
        policy_done,
        policy_cond,
    ) = policy_batch
    (
        expert_obs,
        expert_next_obs,
        expert_action,
        expert_reward,
        expert_done,
        expert_cond,
    ) = expert_batch


    if args.only_expert_states:
        # expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done, expert_cond

    # obs, next_obs, action, reward, done, is_expert = get_concat_samples(
    #     policy_batch, expert_batch, args)
    obs, next_obs, action, reward, done, cond, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    loss_dict = {}

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    # current_Q = self.critic(obs, action)
    # y = (1 - done) * self.gamma * self.getV(next_obs)
    current_Q = self.critic((obs, action, cond))
    y = (1 - done) * self.gamma * self.getV((next_obs, cond))
    if args.train.use_target:
        with torch.no_grad():
            # y = (1 - done) * self.gamma * self.get_targetV(next_obs)
            y = (1 - done) * self.gamma * self.get_targetV((next_obs, cond))

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    # value_loss = (self.getV(obs) - y).mean()
    value_loss = (self.getV((obs,cond)) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss


def iq_update_critic(self, policy_batch, expert_batch, logger, step, cond_type):
    args = self.args
    # policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    # expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done, policy_cond, policy_dist_params = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done, expert_cond, expert_dist_params= expert_batch

    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        # expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done, expert_cond, expert_dist_params

    batch = get_concat_samples(policy_batch, expert_batch, args)
    # obs, next_obs, action = batch[0:3]
    obs, next_obs, action, reward, done, cond, is_expert = batch
    

    agent = self
    # current_V = self.getV(obs)
    if cond_type=="none":
        current_V = self.getV(obs)
    else:
        current_V = self.getV((obs, cond))
    if args.train.use_target:
        with torch.no_grad():
            # next_V = self.get_targetV(next_obs)
            if cond_type=="none":
                next_V = self.get_targetV(next_obs)
            else:
                next_V = self.get_targetV((next_obs, cond))
    else:
        # next_V = self.getV(next_obs)
        if cond_type=="none":
                next_V = self.get_targetV(next_obs)
        else:
            next_V = self.get_targetV((next_obs, cond))
        # next_V = self.getV((next_obs, cond))

    if "DoubleQ" in self.args.q_net._target_:
        # current_Q1, current_Q2 = self.critic(obs, action, both=True)
        if cond_type=="none":
            current_Q1, current_Q2 = self.critic(obs, action, both=True)
        else:
            current_Q1, current_Q2 = self.critic((obs, action, cond), both=True)
        q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch, cond_type)
        q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch, cond_type)
        critic_loss = 1/2 * (q1_loss + q2_loss)

        # additional_loss = "CQL", "current_Q", "combined_loss", "none"
        additional_loss = args.additional_loss
        if additional_loss == "CQL_expertAndPolicy":
            fixed_current_Q = 0
            cql_loss_1 = self.cqlV((obs, cond), self.critic.Q1,args.num_random) - fixed_current_Q
            cql_loss_2 = self.cqlV((obs, cond), self.critic.Q2,args.num_random) - fixed_current_Q
            cql_loss = args.cql_coef*(cql_loss_1+cql_loss_2)/2    
        elif additional_loss == "CQL":
            fixed_current_Q = 0
            cql_loss_1 = self.cqlV((expert_obs, expert_cond), self.critic.Q1,args.num_random) - fixed_current_Q
            cql_loss_2 = self.cqlV((expert_obs, expert_cond), self.critic.Q2,args.num_random) - fixed_current_Q
            cql_loss = args.cql_coef*(cql_loss_1+cql_loss_2)/2    
        elif additional_loss == "currentQ_expertAndPolicy":
            cql_loss = args.cql_coef*(-current_Q1.mean()-current_Q2.mean())/2
        elif additional_loss == "currentQ":
            # Assume policy_batch and expert_batch are tensors or numpy arrays
            # Get the sizes of the policy and expert batches
            policy_batch_size = policy_obs.shape[0]
            expert_batch_size = expert_obs.shape[0]

            # Create is_expert mask: False for policy_batch, True for expert_batch
            is_expert = torch.cat([
                torch.zeros(policy_batch_size, dtype=torch.bool), 
                torch.ones(expert_batch_size, dtype=torch.bool)
            ], dim=0).to(policy_obs.device)
            current_Q1 = current_Q1[~is_expert]
            current_Q2 = current_Q2[~is_expert]

            cql_loss = args.cql_coef*(-current_Q1.mean()-current_Q2.mean())/2    
        elif additional_loss == "combined_expertAndPolicy":
            # policy_batch_size = policy_obs.shape[0]
            # expert_batch_size = expert_obs.shape[0]
            # # Create is_expert mask: False for policy_batch, True for expert_batch
            # is_expert = torch.cat([
            #     torch.zeros(policy_batch_size, dtype=torch.bool), 
            #     torch.ones(expert_batch_size, dtype=torch.bool)
            # ], dim=0).to(policy_obs.device)
            # current_Q1 = current_Q1[~is_expert]
            # current_Q2 = current_Q2[~is_expert]

            cql_loss_1 = self.cqlV((obs, cond), self.critic.Q1,args.num_random) - current_Q1.mean()
            cql_loss_2 = self.cqlV((obs, cond), self.critic.Q2,args.num_random) - current_Q2.mean()
            cql_loss = args.cql_coef*(cql_loss_1+cql_loss_2)/2    
        elif additional_loss == "combined":
            policy_batch_size = policy_obs.shape[0]
            expert_batch_size = expert_obs.shape[0]
            # Create is_expert mask: False for policy_batch, True for expert_batch
            is_expert = torch.cat([
                torch.zeros(policy_batch_size, dtype=torch.bool), 
                torch.ones(expert_batch_size, dtype=torch.bool)
            ], dim=0).to(policy_obs.device)
            current_Q1 = current_Q1[~is_expert]
            current_Q2 = current_Q2[~is_expert]

            cql_loss_1 = self.cqlV((expert_obs, expert_cond), self.critic.Q1,args.num_random) - current_Q1.mean()
            cql_loss_2 = self.cqlV((expert_obs, expert_cond), self.critic.Q2,args.num_random) - current_Q2.mean()
            cql_loss = args.cql_coef*(cql_loss_1+cql_loss_2)/2    
        elif additional_loss == "logsumexp_expertAndPolicy":
            cql_loss_1 = torch.logsumexp(current_Q1, dim=1).mean() * args.cql_coef
            cql_loss_2 = torch.logsumexp(current_Q2, dim=1).mean() * args.cql_coef

            cql_loss_1 -= current_Q1.mean() * args.cql_coef
            cql_loss_2 -= current_Q2.mean() * args.cql_coef
            cql_loss = args.cql_coef*(cql_loss_1+cql_loss_2)/2    
        elif additional_loss == "logsumexp":
            policy_batch_size = policy_obs.shape[0]
            expert_batch_size = expert_obs.shape[0]

            # Create is_expert mask: False for policy_batch, True for expert_batch
            is_expert = torch.cat([
                torch.zeros(policy_batch_size, dtype=torch.bool), 
                torch.ones(expert_batch_size, dtype=torch.bool)
            ], dim=0).to(policy_obs.device)
            current_Q1 = current_Q1[~is_expert]
            current_Q2 = current_Q2[~is_expert]
            
            cql_loss_1 = torch.logsumexp(current_Q1, dim=1).mean() * args.cql_coef
            cql_loss_2 = torch.logsumexp(current_Q2, dim=1).mean() * args.cql_coef

            cql_loss_1 -= current_Q1.mean() * args.cql_coef
            cql_loss_2 -= current_Q2.mean() * args.cql_coef
            cql_loss = args.cql_coef*(cql_loss_1+cql_loss_2)/2    
        else:
            cql_loss = 0

        critic_loss += cql_loss
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
        loss_dict["cql_loss"] = cql_loss
    else:
        # current_Q = self.critic(obs, action)
        if cond_type=="none":
            current_Q = self.critic(obs, action)
        elif args.agent.name=="softq":
            current_Q = self.critic((obs, cond), action)
        else:
            current_Q = self.critic((obs, action, cond))
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch, cond_type)

        additional_loss = args.additional_loss
        if additional_loss != "none":
            raise NotImplementedError("Additional loss not implemented for single Q network")
            policy_batch_size = policy_obs.shape[0]
            expert_batch_size = expert_obs.shape[0]
            # Create is_expert mask: False for policy_batch, True for expert_batch
            is_expert = torch.cat([
                torch.zeros(policy_batch_size, dtype=torch.bool), 
                torch.ones(expert_batch_size, dtype=torch.bool)
            ], dim=0).to(policy_obs.device)
            current_Q = current_Q[~is_expert]

            cql_loss = args.cql_coef*(self.cqlV((expert_obs, expert_cond), self.critic.Q,args.num_random) - current_Q.mean())
            critic_loss += cql_loss
            loss_dict["cql_loss"] = cql_loss
    # logger.log('train/critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()
    return loss_dict


def iq_update(self, policy_buffer, expert_buffer, logger, step, cond_type):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.iq_update_critic(policy_batch, expert_batch, logger, step, cond_type)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:
            if self.args.offline:
                obs = expert_batch[0]
                cond = expert_batch[-2]
                act_demo = expert_batch[2]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)
                cond = torch.cat([policy_batch[-2], expert_batch[-2]], dim=0)
                act_demo = expert_batch[2]
            if not self.args.method.enable_bc_actor_update:
                act_demo = None
            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    # actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
                    if cond_type=="none":
                        actor_alpha_losses = self.update_actor_and_alpha(obs, act_demo, logger, step)
                    else:
                        actor_alpha_losses = self.update_actor_and_alpha((obs,cond), act_demo, logger, step)

            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses

def bc_update(self, observation, action, condition, logger, step, cond_type, mu, log_var, encoder_optimizer, encoder):
    # SAC version
    if self.actor:
        if cond_type == "none":
            training_metrics = self.loss_calculator(self, observation, action)
        else:
            training_metrics = self.loss_calculator(self, (observation, condition), action, mu, log_var)

        loss = training_metrics["loss/bc_actor"]
        # print(training_metrics)

        # optimize actor
        self.actor_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()
        grad_clip_value = 1.0
        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip_value)
        self.actor_optimizer.step()
        encoder_optimizer.step()

        # update critic
        # if step % self.actor_update_frequency == 0:
        #     if cond_type == 'none':
        #         critic_losses = self.bc_update_critic(observation, logger, step)
        #     else:
        #         critic_losses = self.bc_update_critic((observation, condition), logger, step)

        #     training_metrics.update(critic_losses)

        # log
        for key, loss in training_metrics.items():
            if "bc_" in key.split("/")[1]:
                _key = f"train/{key.split('/')[1]}"
            else:
                _key = f"train/bc_{key.split('/')[1]}"
            logger.log(_key, loss, step)

    # Q-Learning version
    else:
        if cond_type == "none":
            training_metrics = self.loss_calculator(self, observation, action)
        else:
            training_metrics = self.loss_calculator(self, (observation, condition), action, mu, log_var)

        loss = training_metrics["loss/bc_actor"]

        # optimize Q function
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # Optionally update target Q networks if using target networks
        if self.use_target_networks:
            self.update_target_networks()

        # log
        for key, loss in training_metrics.items():
            if "bc_" in key.split("/")[1]:
                _key = f"train/{key.split('/')[1]}"
            else:
                _key = f"train/bc_{key.split('/')[1]}"
            logger.log(_key, loss, step)

    # else:
    #     raise NotImplementedError("q learning loop has yet implemented")

    # if step % self.critic_target_update_frequency == 0:
    #     if self.args.train.soft_update:
    #         soft_update(self.critic_net, self.critic_target_net,
    #                     self.critic_tau)
    #     else:
    #         hard_update(self.critic_net, self.critic_target_net)

    return training_metrics



if __name__ == "__main__":
    main()