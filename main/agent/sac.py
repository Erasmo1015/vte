import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra

from utils.utils import soft_update


class SAC(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent

        self.critic_tau = agent_cfg.critic_tau
        self.learn_temp = agent_cfg.learn_temp
        self.actor_update_frequency = agent_cfg.actor_update_frequency
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

        self.critic = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)

        self.critic_target = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=agent_cfg.actor_lr,
                                    betas=agent_cfg.actor_betas)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=agent_cfg.critic_lr,
                                     betas=agent_cfg.critic_betas)
        self.log_alpha_optimizer = Adam([self.log_alpha],
                                        lr=agent_cfg.alpha_lr,
                                        betas=agent_cfg.alpha_betas)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.critic

    @property
    def critic_target_net(self):
        return self.critic_target

    def choose_action(self, state, sample=False):
        # state, cond = state_cond
        # state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # if type(state) == tuple:
        #     state = torch.FloatTensor(state[0]).to(self.device).unsqueeze(0)
        # elif state.ndim == 1:
        #     state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # else:
        #     state = torch.FloatTensor(state).to(self.device)
        # dist = self.actor(state)
        # action = dist.sample() if sample else dist.mean
        # # assert action.ndim == 2 and action.shape[0] == 1
        # return action.detach().cpu().numpy()[0]
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        elif isinstance(state, tuple) or isinstance(state, list):
            #  if state is not tensor then convert it to tensor else keep it that way
            state = [torch.FloatTensor(s).to(self.device).unsqueeze(0) if isinstance(s, np.ndarray) or isinstance(s, list) else s.unsqueeze(0) for s in state]
        # assert len(state)==2
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        # assert action.ndim == 2 and action.shape[0] == 1
        return action.detach().cpu().numpy()[0]

    def getV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        if isinstance(obs, list) or isinstance(obs, tuple):
            _obs, cond = obs
            current_Q = self.critic((_obs, action, cond))
        else:
            current_Q = self.critic(obs, action)
        # current_Q = self.critic(obs, action)
        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V

    def get_targetV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        if isinstance(obs, list) or isinstance(obs, tuple):
            _obs, cond = obs
            target_Q = self.critic_target((_obs, action, cond))
        else:
            target_Q = self.critic_target(obs, action)
        # target_Q = self.critic_target(obs, action)
        target_V = target_Q - self.alpha.detach() * log_prob
        return target_V

    def update(self, replay_buffer, logger, step):
        obs, next_obs, action, reward, done = replay_buffer.get_samples(
            self.batch_size, self.device)

        losses = self.update_critic(obs, action, reward, next_obs, done,
                                    logger, step)

        if step % self.actor_update_frequency == 0:
            actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
            losses.update(actor_alpha_losses)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic, self.critic_target,
                        self.critic_tau)

        return losses

    def update_critic(self, obs, action, reward, next_obs, done, logger, step):

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs)

            target_Q = self.critic_target(next_obs, next_action)
            target_V = target_Q - self.alpha.detach() * log_prob
            target_Q = reward + (1 - done) * self.gamma * target_V

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss = F.mse_loss(current_Q1, target_Q)
        q2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = q1_loss + q2_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(logger, step)
        return {
            'critic_loss/critic_1': q1_loss.item(),
            'critic_loss/critic_2': q2_loss.item(),
            'loss/critic': critic_loss.item()}

    def update_actor_and_alpha(self, obs, act_demo, logger, step):
        action, log_prob, _ = self.actor.sample(obs)
        if isinstance(obs, list) or isinstance(obs, tuple):
            _obs, cond = obs
            actor_Q = self.critic((_obs, action, cond))
            if act_demo is not None:
                act_len = act_demo.shape[0]
                obs_demo, cond_demo = _obs[-act_len:], cond[-act_len:]
                bc_metrics = self.loss_calculator.pure_bc(self, (obs_demo, cond_demo), act_demo)
            else: 
                bc_metrics = None
        else:
            actor_Q = self.critic(obs, action)
            if act_demo is not None:
                act_len = act_demo.shape[0]
                obs_demo = obs[-act_len:]
                bc_metrics = self.loss_calculator.pure_bc(self, obs_demo, act_demo)
            else: 
                bc_metrics = None

        iq_actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        bc_actor_loss = torch.tensor([0.]) if bc_metrics is None else bc_metrics['loss/bc_actor']
        actor_loss = iq_actor_loss + self.bc_alpha * bc_actor_loss if bc_metrics is not None else iq_actor_loss

        neglogp = torch.tensor([0.]) if bc_metrics is None else bc_metrics['bc_actor_loss/neglogp']
        ent_loss = torch.tensor([0.]) if bc_metrics is None else bc_metrics['bc_actor_loss/ent_loss']
        l2_loss = torch.tensor([0.]) if bc_metrics is None else bc_metrics['bc_actor_loss/l2_loss'] 
        
        logger.log('train/actor_loss', actor_loss, step)
        logger.log('train/iq_actor_loss', iq_actor_loss, step)
        logger.log('train/bc_actor_loss', bc_actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/actor_entropy', -log_prob.mean(), step)

        logger.log('train/bc_neglogp', neglogp, step)
        logger.log('train/bc_ent_loss', ent_loss, step)
        logger.log('train/bc_l2_loss', l2_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses = {
            'loss/actor': actor_loss.item(),
            'actor_loss/iq_actor_loss': iq_actor_loss.item(),
            'actor_loss/bc_actor_loss': bc_actor_loss.item(),
            'actor_loss/target_entropy': self.target_entropy,
            'actor_loss/entropy': -log_prob.mean().item()}

        # self.actor.log(logger, step)
        if self.learn_temp:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train/alpha_loss', alpha_loss, step)
            logger.log('train/alpha_value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            losses.update({
                'alpha_loss/loss': alpha_loss.item(),
                'alpha_loss/value': self.alpha.item(),
            })
        return losses

    def evaluate_actions(self, obs, action):
        dist = self.actor(obs)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        if torch.isnan(log_prob).any():
            smallest_value = 1e-6
            action_min = action.min() + smallest_value
            action_max = action.max() - smallest_value
            log_prob = dist.log_prob(torch.clamp(action, action_min, action_max)).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        # check if log_prob is nan
        if isinstance(obs, tuple):
            # Iterate over each element in the tuple
            for idx, obs_e in enumerate(obs):
                if torch.isnan(obs_e).any():
                    print(f'!!!!!!!!!!!!!!! obs element {idx} contains NaN values')
                    print("Obs with nan:",obs_e)
                    raise ValueError("Obs contains NaN values.")
        else:
            # If dist is not a tuple, assume it's already the tensor
            if torch.isnan(obs).any():
                print("Obs with nan:", obs)
                raise ValueError("Obs contains NaN values.")
        if isinstance(action, tuple):
            # Iterate over each element in the tuple
            for idx, i in enumerate(action):
                if torch.isnan(i).any():
                    print(f'!!!!!!!!!!!!!!! action element {idx} contains NaN values')
                    print("Act with nan:", i)
                    raise ValueError("Act contains NaN values.")
        else:
            # If dist is not a tuple, assume it's already the tensor
            if torch.isnan(action).any():
                print("Act with nan:", action)
                raise ValueError("Act contains NaN values.")
        
        if torch.isnan(log_prob).any():
            print('!!!!!!!!!!!!!!! log_prob is nan')
            print("Mean:", dist.mean)
            # print("Standard Deviation:", dist.stddev)  # Ensure stddev > 0
            assert (dist.scale > 0).all(), "Scale must be positive"
            # print("Distribution parameters:", dist.mean, dist.scale)
            print("Distribution parameters:")
            for param_name, param_value in dist.__dict__.items():
                # print(f"{param_name}: {param_value}")
                if isinstance(param_value, torch.Tensor):
                    # Check if param_value tensor contains NaN values
                    if torch.isnan(param_value).any():
                        print(f"!!!!!!!!!!!!!!! {param_name} contains NaN values")
                        print(f"{param_name}: {param_value}")
                # else:
                #     print(f"{param_name}: {param_value} (not a tensor)")
                #     print("param_value: ", param_value)
        return dist, log_prob, entropy

    # Save model parameters
    def save(self, path, suffix=""):
        actor_path = f"{path}{suffix}_actor"
        critic_path = f"{path}{suffix}_critic"

        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        actor_path = f'{path}/{self.args.agent.name}{suffix}_actor'
        critic_path = f'{path}/{self.args.agent.name}{suffix}_critic'
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

    # load from rl_zoo3 baselines
    def load(self, actor_path, critic_path, suffix=""):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device), strict=False)
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device), strict=False)

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()

    def sample_actions(self, obs, num_actions):
        """For CQL style training."""
        obs, cond = obs
        # Process `obs` and `cond` separately
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
            obs.shape[0] * num_actions, obs.shape[1])
        
        cond_temp = cond.unsqueeze(1).repeat(1, num_actions, 1).view(
            cond.shape[0] * num_actions, cond.shape[1])
        
        # Combine them again if needed
        combined_temp = torch.cat([obs_temp, cond_temp], dim=-1)
        # obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
        #     obs.shape[0] * num_actions, obs.shape[1])
        action, log_prob, _ = self.actor.sample((obs_temp, cond_temp))

        return action, log_prob.view(obs.shape[0], num_actions, 1)

    def _get_tensor_values(self, obs, actions, network=None):
        """For CQL style training."""
        obs, cond = obs
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs.shape[0] * num_repeat, obs.shape[1])
        cond_temp = cond.unsqueeze(1).repeat(1, num_repeat, 1).view(
            cond.shape[0] * num_repeat, cond.shape[1])
        obs_cond = torch.cat([obs_temp,cond_temp],dim=-1)
        action_cond = torch.cat([actions, cond_temp], dim=-1)
        obs_cond_action_cond = torch.cat([obs_cond, action_cond], dim=-1)
        preds = network(obs_cond_action_cond)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def cqlV(self, obs, network, num_random=10):
        """For CQL style training."""
        # importance sampled version
        action, log_prob = self.sample_actions(obs, num_random) # policy action
        current_Q = self._get_tensor_values(obs, action, network)
        obs_shape = obs[0].shape[0] # input obs consist of (obs,cond)
        random_action = torch.FloatTensor(
            obs_shape * num_random, action.shape[-1]).uniform_(-1, 1).to(self.device) # random action

        random_density = np.log(0.5 ** action.shape[-1])
        rand_Q = self._get_tensor_values(obs, random_action, network)
        alpha = self.alpha.detach()

        cat_Q = torch.cat(
            [rand_Q - alpha * random_density, current_Q - alpha * log_prob.detach()], 1
        )
        cql_V = torch.logsumexp(cat_Q / alpha, dim=1).mean() * alpha
        return cql_V
