import os
from torch import device
from typing import Tuple, Union
import numpy as np
from d4rl import offline_env
from d4rl.kitchen import kitchen_envs
from gcpc.data.sequence import GOAL_DIMS
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

Device = Union[device, str, int, None]


def get_goal(env: offline_env.OfflineEnv, goal_type: str = 'state', goal_frac=None, obs=None, target_return=None) -> np.ndarray:
    if goal_type == 'state':
        if 'antmaze' in env.spec.id:
            goal = env.target_goal

        elif 'kitchen' in env.spec.id:
            goal = obs[:30].copy()
            subtask_collection = env.TASK_ELEMENTS
            for task in subtask_collection:
                subtask_indices = kitchen_envs.OBS_ELEMENT_INDICES[task]
                subtask_goals = kitchen_envs.OBS_ELEMENT_GOALS[task]
                goal[subtask_indices] = subtask_goals
            goal_mask = np.ones(30, dtype=np.bool8)
            goal_mask[GOAL_DIMS['kitchen']] = False
            goal = np.where(goal_mask, 0., goal)
        else:
            raise NotImplementedError
    else:  # rtg as goal
        # print(f"env max, min score: {env.ref_max_score}, {env.ref_min_score}")
        # max_score = -50 / env._max_episode_steps
        # min_score = -200 / env._max_episode_steps
        max_score = 100000 / env._max_episode_steps
        min_score = 0 / env._max_episode_steps
        # max_score = env.ref_max_score / env._max_episode_steps
        # min_score = env.ref_min_score / env._max_episode_steps
        if target_return is None:
            target_return = max_score - min_score
        goal = min_score + (target_return) * goal_frac

        goal = np.array([goal], dtype=np.float32)
    
    return goal


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)