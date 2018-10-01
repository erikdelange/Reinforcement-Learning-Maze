import logging
import random
from datetime import datetime

import numpy as np

from models import AbstractModel


class QTableModel(AbstractModel):
    """ Prediction model which uses Q-learning and a Q-table.

        For every state (= maze layout with the agents current location ) the Q for each of the actions is stored.
        in a table. The key for this table is (state + action). Initially all Q's are 0. When playing training games
        after every move the Q's in the table are updated based on the reward gained after making the move. Training
        ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).

        :param class Maze game: Maze game object.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.Q = dict()  # table with Q per (state, action) combination

    def train(self, **kwargs):
        """ Hyperparameters:

            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
            :keyword float learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
            :keyword int episodes: number of training games to play
            :return int, datetime: number of training episodes, total time spent
        """
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.99)  # = 1% reduction
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = kwargs.get("episodes", 1000)

        wins = 0
        hist = []  # store evolution of number of wins over the episodes for reporting purposes
        start_list = list()
        start_time = datetime.now()

        for episode in range(1, episodes):
            # optimization: make sure to start from all possible cells
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # change state np.ndarray to tuple so it can be used as dictionary key

            while True:
                # explore less and less as training progresses
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                    exploration_rate *= exploration_decay
                else:
                    action = self.predict(state)

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                if (state, action) not in self.Q.keys():  # ensure a Q exists for (state, action) to avoid a KeyError
                    self.Q[(state, action)] = 0.0

                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])

                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])

                if status in ("win", "lose"):  # terminal state reached, stop training episode
                    if status == "win":
                        wins += 1
                    break

                state = next_state

            hist.append(wins)

            logging.info("episode: {:d}/{:d} | status: {:4s} | total wins: {:d} | e: {:.5f}"
                         .format(episode, episodes, status, wins, exploration_rate))

            if episode % 10 == 0:
                # check if the current model wins from all starting cells
                # can only do this if there is a finite number of starting states
                if self.environment.win_all(self) is True:
                    logging.info("won from all start cells, stop learning")
                    break

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return hist, episode, datetime.now() - start_time

    def predict(self, state):
        """ Policy: choose the action with the highest Q from the Q-table. Random choice if multiple actions
            have the same (max) Q.

            :param np.array state: Game state.
            :return int: Chosen action.
        """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        q = [self.Q.get((state, a), 0.0) for a in self.environment.actions]
        logging.debug("q[] = {}".format(q))

        mv = np.amax(q)  # determine max Q
        actions = np.nonzero(q == mv)[0]  # extract (index of) action(s) with the max Q
        return random.choice(actions)
