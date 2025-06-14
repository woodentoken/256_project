import logging
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from copy import deepcopy

from config.f16_ic_config import ic, type_randomization_variance
from environment.fdm import FDM  # Assuming you have a FlightDynamics class defined
from environment.reward import MaintainFlight  # Assuming you have a RewardFunction class defined

ACTION_SCALING = 1.0

# Configure logging
log_filename = f"logs/train_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
    ],
)


class FDM_env(gym.Env):
    def __init__(self, evaluation=False, randomization_factor=1.0):
        super(FDM_env, self).__init__()
        self.evaluation = evaluation
        self.randomization_factor = randomization_factor  # Default randomization factor
        self.fdm = FDM("f16")
        self.episode_count = -1
        self.last_action = np.zeros(3, dtype=np.float32)
        self.max_action_delta = 0.3  # Maximum change in action per step
        self.logger = logging.getLogger(__name__)

        # Example obs and action spaces (you'll need to define real ones)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def eval_copy(self):
        return FDM_env(
            evaluation=True,
            randomization_factor=0.0
            # etc., depending on your init signature
        )

    def reset(self, *, seed=None, options=None):
        # super().reset(seed=seed)
        self.episode_count += 1
        self.fdm.initialize(deepcopy(ic), randomization_factor=self.randomization_factor)
        self.step_count = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        if self.episode_count % 200 == 0 or self.evaluation:
            print(f"Episode {self.episode_count} reset with randomization factor {self.randomization_factor}")
            self.logger.info(
                f"Episode {self.episode_count} reset with randomization factor {self.randomization_factor}"
            )
            self.state_history = pl.DataFrame()
            self.action_history = pl.DataFrame()
            self.reward_history = pl.DataFrame()

        return self.fdm.get_observation(), {}

    def process_action(self, action):
        action = action * ACTION_SCALING
        delta = np.clip(action - self.last_action, -self.max_action_delta, self.max_action_delta)
        smoothed_action = self.last_action + delta
        self.last_action = smoothed_action.copy()
        return smoothed_action

    def step(self, action):
        self.step_count += 1

        smoothed_action = self.process_action(action)
        self.fdm.set_input(smoothed_action)

        self.fdm.propagate_dynamics()

        obs_dict, full_state = self.fdm.get_state_dict()
        time = full_state["time"]

        # recover the current state from the flight dynamics model
        observation = self.fdm.get_observation()

        # compute the reward based on the current state and action
        reward, constituents = self.get_reward(observation, action, self.step_count)

        if self.episode_count % 200 == 0 or self.evaluation:
            self.state_history = pl.concat([self.state_history, pl.DataFrame([full_state])])
            self.action_history = pl.concat([self.action_history, pl.DataFrame([self.fdm.get_input_dict()])])
            self.reward_history = pl.concat([self.reward_history, pl.DataFrame([constituents])])
            self.logger.info(f"Step {self.step_count}:, Reward: {reward}, Action: {action}, Observation: {obs_dict}")

        # determine if the episode is done
        terminated, truncated = self.check_done(observation, self.step_count)

        # additional info can be returned, e.g., for logging
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "episode_count": self.episode_count,
        }

        if terminated or truncated:
            if self.episode_count % 200 == 0 or self.evaluation:
                self.logger.info(f"Episode ended: Terminated: {terminated}, Truncated: {truncated}, Time: {time}\n\n\n")
                # save the state and action history to a pickle file
                with open(f"logs/state_action_reward_history_episode_{self.episode_count}.pkl", "wb") as f:
                    pickle.dump(
                        {"state": self.state_history, "action": self.action_history, "reward": self.reward_history}, f
                    )
            info["episode/truncated"] = truncated
            info["episode/terminated"] = terminated
            info["episode/altitude"] = float(observation[0])
            info["episode/airspeed"] = float(observation[1])
            info["episode/total_reward"] = float(reward)

        return observation, reward, terminated, truncated, info

    def get_reward(self, observation, action, step_count):
        reward = MaintainFlight().get_reward(observation, action, step_count)
        return reward

    def check_done(self, observation, step_count):
        terminated = False
        truncated = False

        if observation[0] < 20 or observation[0] > 1000000:
            terminated = True

        if step_count > 66666:
            truncated = True

        return terminated, truncated

    def render(self, mode="human"):
        pass
