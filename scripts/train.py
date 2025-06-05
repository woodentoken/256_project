import yaml
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.fdm_env import FDM_env  # Assuming you have a custom environment defined in jsbsim_env.py

def train(algo, subconfig):
    with open("config/ppo_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ppo_kwargs = config.get(subconfig, {})
    print(ppo_kwargs)

    env = DummyVecEnv(
        [lambda: Monitor(FDM_env(), filename=f"{subconfig}_log.csv", info_keywords=("terminated","truncated", "episode_count"))]  # Wrap the environment in a Monitor for logging
    )  # Wrap the environment in a DummyVecEnv for vectorized trainin
    env = VecNormalize(env, norm_obs=True, norm_reward=True)  # Normalize observations and rewards

    obs = env.reset()  # Reset the environment to get the initial observation
    print("Initial observation:", obs)

    ppo_model = algo("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_jsbsim_tensorboard/", **ppo_kwargs)

    ppo_model.learn(total_timesteps=300_000)  # Adjust the number of timesteps as needed
    ppo_model.save("models/ppo_jsbsim")  # Save the trained model
    print("Training complete. Model saved as 'ppo_jsbsim'.")

    env.save(f"models/{subconfig}_jsbsim_normalize.pkl")  # Save the VecNormalize statistics

if __name__ == "__main__":
    train(algo=PPO, subconfig="ppo")
    train(algo=PPO, subconfig="ppo_fast_learner")
    train(algo=PPO, subconfig="ppo_low_gamma")
    train(algo=PPO, subconfig="ppo_fast_learner_low_gamma")
