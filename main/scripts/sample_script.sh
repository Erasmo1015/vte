#!/usr/bin/env bash

# Run conditional imitation leaerning task 

# Set working directory to main
cd ..

# Hopper-v2
python train_iq.py env.cond=hopper/[traj embedding file] exp_dir=[encoder directory absolute path] encoder=[encoder file] env=hopper env.learn_steps=1e6 cond_dim=10 method.kld_alpha=1 agent.actor_lr=3e-05 agent.init_temp=1e-12 wandb=True agent=sac expert.demos=30 method.enable_bc_actor_update=False method.bc_init=False method.bc_alpha=0.5 env.eval_interval=1e4 cond_type=debug env.demo=hopper/Hopper-v2_30_409r+876r+3213r.pkl method.loss=v0 method.regularize=True num_levels=3 additional_loss=none seed=1

# HalfCheetah-v2
python train_iq.py env.cond=cheetah/[traj embedding file] exp_dir=[encoder directory absolute] env=cheetah path encoder=[encoder file] additional_loss=none cql_coef=1 num_random=5 env.learn_steps=3e6 cond_dim=10 method.kld_alpha=10 agent.actor_lr=5e-5 agent.critic_lr=1e-4 agent.init_temp=1e-12 seed=0 wandb=True agent=sac expert.demos=30 method.enable_bc_actor_update=False method.bc_init=False method.bc_alpha=0.5 env.eval_interval=1e4 cond_type=debug env.demo=cheetah/HalfCheetah-v2_30_2402r+4208r+6301r.pkl method.loss=value method.regularize=True num_levels=3 save_last=True save_interval=5000 env.eval_interval=2e4 seed=1