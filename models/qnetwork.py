import logging
import random

import numpy as np
from keras import Sequential
from keras.layers import Dense

from game import actions
from models import AbstractModel


class QNetworkModel(AbstractModel):
    """ Prediction model which uses Q-learning and a simple neural network.

        The network learns by playing training games. After every move the Q's are updated according to
        the Bellman equation. The resulting state + Q's are fed into the network. The state is represented
        as a [1][N] vector where N is the number of cells in the maze. The training algorithm ensures that
        the game is started from every possible cell. Training ends after a fixed number of games, or
        earlier if a stopping criterion is reached (here: a 100% win rate).

        :param class Maze game: Maze game object.
    """

    def __init__(self, game):
        super().__init__(game)

        self.model = Sequential()
        self.model.add(Dense(game.maze.size, input_shape=(game.maze.size,), activation="sigmoid"))
        self.model.add(Dense(game.maze.size, activation="sigmoid"))
        self.model.add(Dense(len(actions), activation="linear"))
        self.model.compile(optimizer="adam", loss="mse")

    def train(self, **kwargs):
        """ Hyperparameters:

            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword int episodes: number of training games to play
        """

        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        episodes = kwargs.get("episodes", 1000)

        wins = 0
        start_list = list()  # starting cells not yet used for training

        for episode in range(1, episodes):
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)

            while True:
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = np.argmax(self.model.predict(state))

                next_state, reward, status = self.environment.step(action)

                if status in ("win", "lose"):
                    target = reward  # no discount needed if a terminal state was reached.
                else:
                    target = reward + discount * np.max(self.model.predict(next_state))

                target_vector = self.model.predict(state)
                target_vector[0][action] = target  # update Q value for this action

                self.model.fit(state, target_vector, epochs=4, verbose=0)
                # loss += self.models.evaluate(state, target_vector, verbose=0)

                if status in ("win", "lose"):  # terminal state reached, stop episode
                    if status == "win":
                        wins += 1
                    break

                state = next_state

            logging.info("episode: {:05d}/{:05d} | status: {:4s} | total wins: {:4d} ({:.2f})"
                         .format(episode, episodes, status, wins, wins / episode))

            if episode % 10 == 0:
                # check if current model wins from all starting cells
                # can only do this if there is a finite number of starting states
                if self.environment.win_all(self) is True:
                    logging.info("Won from all start cells, exit learning")
                    break

    def predict(self, state):
        """ Choose the action with the highest Q from the Q network.

            :param np.array state: Game state.
            :return int: Chosen action.
        """
        q = self.model.predict(state)
        return np.argmax(q[0])  # action is the index of the highest Q value
