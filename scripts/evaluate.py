import os
import random
import sys

import numpy as np
import torch
from icecream import ic
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.fdm_env import FDM_env
from utils.plotting import plot_path, plot_trajectory

# change model meta to the model you want to evaluate
model_meta = "a=0.0002, gamma=0.99"
SEED = 666

np.random.seed(SEED)  # Set the random seed for reproducibility
torch.manual_seed(SEED)  # Set the random seed for PyTorch
random.seed(SEED)  # Set the random seed for the random module


def make_env(seed, eval=False):
    def _init():
        env = FDM_env(evaluation=eval, randomization_factor=0.0)  # Create the environment
        # env.reset(seed=seed)  # Optional but helpful to trigger RNG
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    return _init


# Create a vectorized environment for evaluation
vec_eval_env = DummyVecEnv([make_env(SEED, eval=True)])  # 1 env for evaluation
vecnorm_eval_env = VecNormalize.load(
    f"models/{model_meta}_normalize.pkl", vec_eval_env
)  # Load the normalization statistics
vecnorm_eval_env.training = False  # Set the environment to evaluation mode
vecnorm_eval_env.norm_reward = False  # Disable reward normalization for evaluation
vecnorm_eval_env.seed(SEED)  # Set the seed for reproducibility
vecnorm_eval_env.venv.seed(SEED)  # Set the seed for the vectorized environment
inner_env = vecnorm_eval_env.venv.envs[0]  # Get the inner environment
# inner_env.randomization_factor = 0  # Set the randomization factor to 0 for evaluation

model = PPO.load(
    f"models/{model_meta}_best/best_model", env=vecnorm_eval_env, device="cpu"
)  # Load the trained PPO model

# Load normalization statistics
obs = vecnorm_eval_env.reset()  # Reset the environment to get the initial observation

done = False
while not done:
    sh = inner_env.state_history
    ah = inner_env.action_history
    rh = inner_env.reward_history

    action, states = model.predict(obs, deterministic=True)  # Predict the action using the model
    obs, rewards, done, info = vecnorm_eval_env.step(action)

plot_trajectory(sh, ah, rh)
plot_path(sh, interactive=False)
