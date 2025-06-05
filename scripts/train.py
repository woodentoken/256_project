import yaml
from environment.simulation import FDM_env  # Assuming you have a custom environment defined in jsbsim_env.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agents.ppo_agent import create_agent  # Assuming you have a function to create the agent


def main():
    with open("configs/ppo_config.yaml", "r") as f:
        config = yaml.load(f)

    env = FDM_env()
    check_env(env)  # Check if the environment follows the Gym API

    agent = create_agent(env, config)

    agent.learn(total_timesteps=10000)  # Adjust the number of timesteps as needed
    agent.save("ppo_jsbsim")  # Save the trained model
    print("Training complete. Model saved as 'ppo_jsbsim'.")


if __name__ == "__main__":
    main()
