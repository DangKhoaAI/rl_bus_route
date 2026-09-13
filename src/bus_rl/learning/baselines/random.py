import numpy as np


class RandomValidController:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation, mask):
        return int(self.rng.choice(np.flatnonzero(mask)))
