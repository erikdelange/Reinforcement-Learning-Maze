import logging
import random
from datetime import datetime

import numpy as np
from keras import Sequential
from keras.layers import Dense

from environment.maze import actions
from models import AbstractModel


class QNetworkModel(AbstractModel):
    """ Prediction model which uses Q-learning and a simple neural network.

        The network learns how states connect to actions by playing training games. After every move the Q's
        are updated according to the Bellman equation. The resulting state + Q's are fed into the network.
        State is represented as a [1][N] vector where N is the number of cells in the maze. The training
        algorithm ensures that the game is started from every possible cell. Training ends after a fixed
        number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).

        :param class Maze game: Maze game object.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)

        self.model = Sequential()
        self.model.add(Dense(game.maze.size, input_shape=(game.maze.size,), activation="relu"))
        self.model.add(Dense(game.maze.size, activation="relu"))
        self.model.add(Dense(len(actions), activation="linear"))
        self.model.compile(optimizer="adam", loss="mse")

    def train(self, **kwargs):
        """ Hyperparameters:

            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword int episodes: number of training games to play
            :return int, datetime: number of training episodes, total time spent
        """

        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        episodes = kwargs.get("episodes", 1000)

        wins = 0
        hist = []
        start_list = list()  # starting cells not yet used for training
        start_time = datetime.now()

        for episode in range(1, episodes):
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)

            loss = 0.0

            while True:
                q = self.model.predict(state)

                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    mv = np.amax(q[0])
                    actions = np.nonzero(q[0] == mv)[0]
                    action = random.choice(actions)

                next_state, reward, status = self.environment.step(action)

                if status in ("win", "lose"):
                    target = reward  # no discount needed if a terminal state was reached.
                else:
                    target = reward + discount * np.amax(self.model.predict(next_state)[0])

                q[0][action] = target  # update Q value for this action

                self.model.fit(state, q, epochs=1, verbose=0)
                loss += self.model.evaluate(state, q, verbose=0)

                if status in ("win", "lose"):  # terminal state reached, stop episode
                    if status == "win":
                        wins += 1
                    break

                state = next_state

            hist.append(wins)

            logging.info("episode: {:d}/{:d} | status: {:4s} | loss: {:.4f} | total wins: {:d} | e: {:.5f}"
                         .format(episode, episodes, status, loss, wins, exploration_rate))

            if episode % 10 == 0:
                # check if the current model wins from all starting cells
                # can only do this if there is a finite number of starting states
                if self.environment.win_all(self) is True:
                    logging.info("won from all start cells, stop learning")
                    break

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return hist, episode, datetime.now() - start_time

    def predict(self, state):
        """ Policy: choose the action with the highest Q from the Q-table. Random choice if there are multiple actions
            with an equal max Q.

            :param np.array state: Game state.
            :return int: Chosen action.
        """
        q = self.model.predict(state)

        mv = np.amax(q[0])  # determine max Q
        actions = np.nonzero(q[0] == mv)[0]  # extract (index of) action(s) with the max Q
        return random.choice(actions)
