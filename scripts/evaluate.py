from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plotting import plot_trajectory, plot_path
from environment.fdm_env import FDM_env

eval_env = DummyVecEnv([lambda: FDM_env()])  # Create a vectorized environment for evaluation

eval_env = VecNormalize.load("models/ppo_normalize.pkl", eval_env)  # Load the normalization statistics
eval_env.training = False  # Set the environment to evaluation mode
eval_env.norm_reward = False  # Disable reward normalization for evaluation

model = PPO.load("models/ppo", env=eval_env, device="cpu")  # Load the trained PPO model

# Load normalization statistics
obs = model.env.reset()

done = False
while not done:
    action, states = model.predict(obs)
    obs, rewards, done, info = eval_env.step(action)

sh = eval_env.envs[0].state_history
ah = eval_env.envs[0].action_history
rh = eval_env.envs[0].reward_history

plot_trajectory(sh, ah, rh)
plot_path(sh, interactive=False)