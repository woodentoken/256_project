class RewardFunction:
    def __init__(self, config):
        self.config = config

    def get_reward(self, state, action, next_state):
        raise NotImplementedError("This method should be overridden by subclasses")


    def reset(self):
        pass

    def update(self, state, action, next_state, reward):
        pass