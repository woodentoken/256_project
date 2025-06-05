from stable_baselines3 import PPO
from environment.fdm_env import FDM_env

env = FDM_env()
model = PPO.load("ppo_jsbsim")
obs = model.env.reset()

done = False
while not done:
    action, states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # model.env.render()  # If you have a render method in your environment