# class RewardFunction:
#     def __init__(self):
#         pass

#     def get_reward(self, state, action, next_state):
#         raise NotImplementedError("This method should be overridden by subclasses")
import numpy as np

class MaintainFlight():
    def __init__(self):
        self.prev_action = np.zeros(3)

    def get_reward(self, observation, action, step_count):
        preservation_bonus = 0.1 + 0.1*(step_count/10)  # assuming a baseline reward for maintaining flight

        # Exponential altitude penalty that increases as aircraft gets closer to ground
        # Assuming observation[0] is altitude in feet, with ground at 0 feet
        altitude = max(0, observation[0])  # Ensure non-negative altitude
        if altitude < 5000:  # Apply penalty below 5000 feet
            collision_penalty = -10 * np.exp(-altitude / 1000)  # Exponential increase as altitude decreases
        else:
            collision_penalty = 0.0

        # Control smoothness penalty (L2 norm of action difference)
        control_delta = np.linalg.norm(action - self.prev_action)
        smoothness_bonus = -np.log(1 + control_delta)  # logarithmic penalty for control changes
        self.prev_action = action.copy()

        # penalize large control inputs
        control_penalty = -0.33*sum(abs(action))**2  # penalize large control inputs

        # attitude_penalty = 0.0
        # if abs(observation[4]) > 90:
        #     attitude_penalty = -np.sqrt(np.abs((observation[4] - 90)))

        total = preservation_bonus + smoothness_bonus + collision_penalty + control_penalty
        constituents = {
            "total": float(total),
            "preservation_bonus": float(preservation_bonus),
            "smoothness_bonus": float(smoothness_bonus),
            "collision_penalty": float(collision_penalty),
            "control_penalty": float(control_penalty),
            # "attitude_penalty": float(attitude_penalty)
        }
        return total, constituents