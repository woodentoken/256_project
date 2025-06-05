import gym
from gym import spaces
import numpy as np
import polars as pl
import jsbsim
from jsb_flight_dynamics import FlightDynamics  # Assuming you have a FlightDynamics class defined
from reward import RewardFunction  # Assuming you have a RewardFunction class defined

class FDM_env(gym.Env):
    def __init__(self):
        super(FDM_env, self).__init__()
        self.fdm = FlightDynamics()

        # Example obs and action spaces (you'll need to define real ones)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        # self.fdm.load_ic("reset00")  # or manually reset the sim state
        self.fdm.initialize(randomization_factor=0.0)
        return self.fdm.get_state()

    def step(self, action):
        # apply the input action to the flight dynamics model
        self.fdm.set_input(action)

        # propagate the flight dynamics model for one time step
        self.fdm.propagate_dynamics()
        
        # recover the current state from the flight dynamics model
        state = self.fdm.get_state()
        
        # compute the reward based on the current state
        reward = self.compute_reward(state)
        
        # determine if the episode is done
        done = self.check_done()
        
        return state, reward, done, {}

    def compute_reward(self, state):
        reward = RewardFunction.get_reward(state, config=None)  # Pass your config
        return reward  # Example: penalize pitch deviation

    def check_done(self):
        return False  # Add custom conditions

    def render(self, mode="human"):
        pass
