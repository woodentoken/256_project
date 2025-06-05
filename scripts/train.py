import yaml
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.fdm_env import FDM_env  # Assuming you have a custom environment defined in jsbsim_env.py

def main():
    with open("config/ppo_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ppo_kwargs = config.get("ppo", {})
    print(ppo_kwargs)

    # raw_env = FDM_env()  # Create an instance of your custom environment
    # check_env(raw_env)  # Check if the environment follows the Gym API

    env = DummyVecEnv(
        [lambda: Monitor(FDM_env(), filename="ppo_log.csv", info_keywords=("terminated","truncated", "episode_count"))]  # Wrap the environment in a Monitor for logging
    )  # Wrap the environment in a DummyVecEnv for vectorized trainin
    env = VecNormalize(env, norm_obs=True, norm_reward=True)  # Normalize observations and rewards

    obs = env.reset()  # Reset the environment to get the initial observation
    print("Initial observation:", obs)

    ppo_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_jsbsim_tensorboard/", **ppo_kwargs)

    ppo_model.learn(total_timesteps=1_000_000)  # Adjust the number of timesteps as needed
    ppo_model.save("models/ppo_jsbsim")  # Save the trained model
    print("Training complete. Model saved as 'ppo_jsbsim'.")

    env.save("models/ppo_jsbsim_normalize.pkl")  # Save the VecNormalize statistics


if __name__ == "__main__":
    main()
