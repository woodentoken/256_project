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
        baseline = 0.1

        # altitude_penalty = 0.2*(observation[0] - 10000)  # assuming observation[0] is altitude in feet

        # Control smoothness penalty (L2 norm of action difference)
        control_delta = np.linalg.norm(action - self.prev_action)
        smoothness_penalty = -control_delta  # adjust coefficient

        w_penalty = observation[3]  # assuming observation[2] is vertical speed (w velocity)

        self.prev_action = action.copy()

        # penalize large control inputs
        control_penalty = -sum(10*abs(action))**2  # penalize large control inputs

        return baseline + control_penalty + smoothness_penalty + w_penalty