"""
abstract class for each environment
"""

class Env_Base:
    def __init__(self):
        pass

    def get_agents_info(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
