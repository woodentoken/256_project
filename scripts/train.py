import os
import sys

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.fdm_env import FDM_env  # Assuming you have a custom environment defined in jsbsim_env.py


def train(algo, subconfig):
    with open("config/ppo_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ppo_kwargs = config.get(subconfig, {})
    print(ppo_kwargs)

    train_env = DummyVecEnv(
        [
            lambda: Monitor(
                FDM_env(randomization_factor=2.0),
                filename=f"training_logs/{subconfig}_log.csv",
                info_keywords=("terminated", "truncated", "episode_count"),
            )
        ]  # Wrap the environment in a Monitor for logging
    )  # Wrap the environment in a DummyVecEnv for vectorized trainin
    env = VecNormalize(train_env, norm_obs=True, norm_reward=True)  # Normalize observations and rewards

    eval_env = DummyVecEnv([lambda: FDM_env()])  # Create a vectorized environment for evaluation
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, training=False
    )  # Normalize observations but not rewards for evaluation

    # Define the callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{subconfig}_best/",
        log_path="./logs/",
        eval_freq=20000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    env.reset()  # Reset the environment to get the initial observation
    ppo_model = algo("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_jsbsim_tensorboard/", **ppo_kwargs)
    ppo_model.learn(
        total_timesteps=1_000_000, callback=eval_callback, tb_log_name=subconfig
    )  # Adjust the number of timesteps as needed

    env.save(f"models/{subconfig}_normalize.pkl")  # Save the VecNormalize statistics
    print(f"Training complete. Model saved as '{subconfig}.zip'.")


if __name__ == "__main__":
    # get the top level configs from the config file
    with open("config/ppo_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    top_level_configs = list(config.keys())

    for subconfig in top_level_configs:
        train(algo=PPO, subconfig=subconfig)
