from stable_baselines3 import PPO
from environment.fdm_env import FDM_env

env = FDM_env()
model = PPO.load("ppo_jsbsim")

# Load normalization statistics
env.load_normalization("models/ppo_jsbsim_normalize.pkl")

obs = model.env.reset()

done = False
while not done:
    action, states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

