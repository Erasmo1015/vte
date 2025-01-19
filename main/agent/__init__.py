import gym
from omegaconf import OmegaConf

from agent.sac import SAC
from agent.softq import SoftQ


def make_agent(env, args):
    # there is args.env.obs_dim, give it to obs_dim
    if hasattr(args.env, 'obs_dim'):
        obs_dim = args.env.obs_dim
    else:
        obs_dim = env.observation_space.shape[0]
    cond_dim = args.cond_dim
    cond_type = args.cond_type

    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        print('--> Using Soft-Q agent')
        action_dim = env.action_space.n
        if cond_type!="none":
            # change the q_net to the conditional version
            print('--> Using conditional Soft-Q agent')
            OmegaConf.set_struct(args, False)
            args.q_net._target_ = 'agent.softq_models.CondOfflineQNetwork'
            args.q_net.cond_dim = cond_dim
            OmegaConf.set_struct(args, True)
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = SoftQ(obs_dim, action_dim, args.train.batch, args)
    else:
        print('--> Using SAC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        if cond_type!="none":
            # change the actor/critic to the conditional version
            print('--> Using conditional SAC agent')
            OmegaConf.set_struct(args, False) # allow dynamic attribute assignment
            args.diag_gaussian_actor._target_ = 'agent.sac_models.CondDiagGaussianActor'
            args.diag_gaussian_actor.cond_dim = cond_dim
            args.q_net._target_ = 'agent.sac_models.CondDoubleQCritic'
            args.q_net.cond_dim = cond_dim
            OmegaConf.set_struct(args, True)

        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = SAC(obs_dim, action_dim, action_range, args.train.batch, args)

    return agent
