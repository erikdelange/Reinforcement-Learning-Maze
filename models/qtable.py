import logging
import random

import numpy as np

from game import CELL_CURRENT
from models import AbstractModel


class QTableModel(AbstractModel):
    """ Prediction model which uses Q-learning and a Q-table.

        For every state (= maze layout with the agents current location ) the Q for each of the actions is stored.
        Initially all Q's are 0. When playing training games after every move the Q's are updated according to
        the Bellman equation. The training algorithm ensures that the game is started from every possible cell.
        Training ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a
        100% win rate).

        :param class Maze game: Maze game object.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.qtable = dict()

        for cell in game.cells:
            state = np.copy(self.environment.maze)
            state[cell[::-1]] = CELL_CURRENT
            state = tuple(state.flatten())  # convert [1][Z] array to tuple([Z]) so it can be used as dictionary key
            self.qtable[state] = [0, 0, 0, 0]  # 4 possible actions

    def train(self, **kwargs):
        """ Hyperparameters:

            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
            :keyword int episodes: number of training games to play
        """
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = kwargs.get("episodes", 1000)

        wins = 0
        start_list = list()  # starting cells not yet used for training

        for episode in range(1, episodes):
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())

            while True:
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = np.argmax(self.qtable[state])

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                self.qtable[state][action] += learning_rate * (
                        reward + discount * max(self.qtable[next_state]) - self.qtable[state][action])

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
                    logging.info("Won from all start cells, stop learning")
                    break

    def predict(self, state):
        """ Choose the action with the highest Q from the Q-table.

        :param np.array state: Game state (= index in the Q table).
        :return int: Chosen action.
        """
        state = tuple(state.flatten())
        q = self.qtable[state]
        return np.argmax(q)  # action is the index of the highest Q value
