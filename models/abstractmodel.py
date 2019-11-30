""" Abstract base class for prediction models.
"""
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, maze, **kwargs):
        self.environment = maze
        self.name = kwargs.get("name", "model")

    def load(self, filename):
        """ Load model from file. """
        pass

    def save(self, filename):
        """ Save model to file. """
        pass

    def train(self, stop_at_convergence=False, **kwargs):
        """ Train model. """
        pass

    @abstractmethod
    def q(self, state):
        """ Return q values for state. """
        pass

    @abstractmethod
    def predict(self, state):
        """ Predict value based on state. """
        pass
