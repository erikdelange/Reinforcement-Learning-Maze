import logging
import random
from datetime import datetime

import numpy as np

from environment.maze import CELL_CURRENT
from models import AbstractModel


class QTableTraceModel(AbstractModel):
    """ Prediction model which uses Q-learning and uses a Q-table and an eligibility trace.

        For every state (= maze layout with the agents current location ) the Q for each of the actions is stored.
        in a table. Initially all Q's are 0. When playing training games after every move the Q's in the table are
        updated according to the Bellman equation (= based on the reward gained after making the move). Training
        ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).
        The model keeps track of the states which have been visited and also updates the Q's of previous
        state-action pairs based on the current reward (a.k.a. eligibility trace). With every step the amount
        in which previous Q's are update decays. This approach is meant to speed up learning.

        :param class Maze game: Maze game object.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.game = game
        self.qtable = dict()

        for cell in game.cells:
            state = np.copy(self.environment.maze)
            state[cell[::-1]] = CELL_CURRENT
            state = tuple(state.flatten())  # convert [1][N] array to tuple([N]) so it can be used as dictionary key
            self.qtable[state] = [0, 0, 0, 0]  # 4 possible actions

    def train(self, **kwargs):
        """ Hyperparameters:

            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
            :keyword int episodes: number of training games to play
            :return int, datetime: number of training episodes, total time spent
        """
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = kwargs.get("episodes", 1000)

        wins = 0
        hist = []  # store evolution of number of wins over the episodes
        decay = 0.75  # eligibility trace decay rate
        start_list = list()
        start_time = datetime.now()

        for episode in range(1, episodes):
            etrace = dict()

            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)
            # start_cell = random.choice(self.environment.empty)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())

            while True:
                try:
                    etrace[state] += 1
                except KeyError:
                    etrace[state] = 1

                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = np.argmax(self.qtable[state])

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                # update Q's in trace
                delta = reward + discount * max(self.qtable[next_state]) - self.qtable[state][action]

                for key in etrace.keys():
                    self.qtable[key][action] += learning_rate * delta * etrace[key]

                # decay eligibility trace
                for key in etrace.keys():
                    etrace[key] *= (discount * decay)

                if status in ("win", "lose"):  # terminal state reached, stop episode
                    if status == "win":
                        wins += 1
                    break

                state = next_state

            hist.append(wins)

            logging.info("episode: {:d}/{:d} | status: {:4s} | total wins: {:d}"
                         .format(episode, episodes, status, wins))

            if episode % 10 == 0:
                # check if the current model wins from all starting cells
                # can only do this if there is a finite number of starting states
                if self.environment.win_all(self) is True:
                    logging.info("won from all start cells, stop learning")
                    break

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return hist, episode, datetime.now() - start_time

    def predict(self, state):
        """ Choose the action with the highest Q from the Q-table. Also called the policy.

            :param np.array state: Game state (= index in the Q table).
            :return int: Chosen action.
        """
        state = tuple(state.flatten())
        q = self.qtable[state]

        logging.debug("q[] = {}".format(q))

        return np.argmax(q)  # action is the index of the highest Q value
