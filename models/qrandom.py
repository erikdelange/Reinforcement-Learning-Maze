import random

import numpy as np

from models import AbstractModel


class RandomModel(AbstractModel):
    """ Prediction model which randomly chooses the next action. """

    def __init__(self, game, **kwargs):
        super().__init__(game, name="RandomModel", **kwargs)

    def q(self, state):
        """ Return Q value for all actions for a certain state.

            :return np.ndarray: Q values
        """
        return np.array([0, 0, 0, 0])

    def predict(self, **kwargs):
        """ Randomly choose the next action.

            :return int: selected action
        """
        return random.choice(self.environment.actions)
