from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.gym_env import GymEnv

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from rllab.envs.mujoco.ant_env import AntEnv

env = normalize(SwimmerEnv())
# env = normalize(AntMazeEnv())
# env = normalize(GymEnv("FetchReach-v1"))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(64, 64)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=2500,
    max_path_length=100,
    n_itr=100,
    discount=0.995,
    step_size=0.01,
)
algo.train()
