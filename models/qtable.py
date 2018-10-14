import logging
import random
from datetime import datetime

import numpy as np

from models import AbstractModel


class QTableModel(AbstractModel):
    """ Tabular Q-learning prediction model.

        For every state (= maze layout with the agents current location ) the value for each of the actions is stored.
        in a table. The key for this table is (state + action). Initially all values are 0. When playing training games
        after every move the value in the table is updated based on the reward gained after making the move. Training
        ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).

        :param class Maze game: Maze game object.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.Q = dict()  # table with value for (state, action) combination

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
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # reduction per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = kwargs.get("episodes", 1000)

        # variables for reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()
        start_time = datetime.now()

        for episode in range(1, episodes + 1):
            # optimization: make sure to start from all possible cells
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # change np.ndarray to tuple so it can be used as dictionary key

            while True:
                # choose action epsilon greedy (off-policy, instead of only using the learned policy)
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = self.predict(state)

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                cumulative_reward += reward

                if (state, action) not in self.Q.keys():  # ensure value exists for (state, action) to avoid a KeyError
                    self.Q[(state, action)] = 0.0

                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])

                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])

                if status in ("win", "lose"):  # terminal state reached, stop training episode
                    break

                state = next_state

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status, exploration_rate))

            if episode % 5 == 0:
                # check if the current model wins from all starting cells
                # can only do this if there is a finite number of starting states
                w_all, win_rate = self.environment.win_all(self)
                win_history.append((episode, win_rate))
                # if w_all is True:
                #    logging.info("won from all start cells, stop learning")
                #    break

            exploration_rate *= exploration_decay  # explore less as training progresses

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: Game state.
            :return int: Chosen action.
        """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        q = [self.Q.get((state, a), 0.0) for a in self.environment.actions]
        logging.debug("q[] = {}".format(q))

        mv = np.amax(q)  # determine max value
        actions = np.nonzero(q == mv)[0]  # get index of the action(s) with the max value
        return random.choice(actions)
