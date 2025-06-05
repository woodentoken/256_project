from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def create_agent(env, config):
    """
    Create a PPO agent for the given environment.

    Args:
        env: The environment to train the agent on.
        config: Configuration parameters for the agent.

    Returns:
        A PPO agent instance.
    """
    # Create a vectorized environment
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Create the PPO agent
    agent = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./logs/tensorboard", **config)

    return agent