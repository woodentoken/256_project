import numpy as np

class MaintainFlight():
    def __init__(self):
        self.prev_action = np.zeros(3)

    def get_reward(self, observation, action, step_count):
        altitude = max(0, observation[0])  # Ensure non-negative altitude
        preservation_bonus = 0.01 + 0.05*step_count + 0.1 * (altitude / 5000)  # Reward for maintaining altitude above 1000 feet  # assuming a baseline reward for maintaining flight

        # Control smoothness penalty (L2 norm of action difference)
        smoothness_penalty = -3*np.sum((action - self.prev_action) ** 2) # logarithmic penalty for control changes
        self.prev_action = action.copy()

        # penalize large control inputs
        control_penalty = -0.3*np.sum(np.square(action))  # penalize large control inputs

        if np.abs(observation[4]) > 120:  # assuming observation[1] is pitch angle in degrees
            roll_penalty = -0.01 * (np.abs(observation[1]) - 120)  # penalize large roll angles
        else:
            roll_penalty = 0.001

        if step_count == 1:
            initial_penalty = -1000
        else:
            initial_penalty = 0

        total = preservation_bonus + smoothness_penalty + control_penalty + roll_penalty + initial_penalty
        constituents = {
            "total": float(total),
            "preservation_bonus": float(preservation_bonus),
            "smoothness_penalty": float(smoothness_penalty),
            "control_penalty": float(control_penalty),
            "roll_penalty": float(roll_penalty),
        }
        return total, constituents