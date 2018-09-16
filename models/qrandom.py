import random

from models import AbstractModel


class RandomModel(AbstractModel):
    """ Prediction model which randomly chooses the next action. """

    def __init__(self, game):
        super().__init__(game)

    def predict(self, **kwargs):
        """ Randomly choose the next action.

            :return int: Chosen action.
        """
        return random.choice(self.environment.actions)
