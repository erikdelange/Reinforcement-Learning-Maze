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
        are updated. The resulting state + Q's are fed into the network.
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

    def train(self, stop_at_convergence=False, **kwargs):
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

        start_list = list()  # starting cells not yet used for training
        start_time = datetime.now()

        for episode in range(1, episodes + 1):
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

                cumulative_reward += reward

                if status in ("win", "lose"):
                    target = reward  # no discount needed if a terminal state was reached.
                else:
                    max_next_Q = np.amax(self.model.predict(next_state)[0])
                    target = reward + discount * max_next_Q

                # q[0][action] += learning_rate * (target - q[0][action])  # update Q value for this action
                q[0][action] = target  # update Q value for this action

                self.model.fit(state, q, epochs=1, verbose=0)
                loss += self.model.evaluate(state, q, verbose=0)

                if status in ("win", "lose"):  # terminal state reached, stop episode
                    break

                state = next_state

                self.environment.render_q(self)

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | loss: {:.4f} | e: {:.5f}"
                         .format(episode, episodes, status, loss, exploration_rate))

            if episode % 5 == 0:
                # check if the current model wins from all starting cells
                # can only do this if there is a finite number of starting states
                w_all, win_rate = self.environment.win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            exploration_rate *= exploration_decay  # explore less as training progresses

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        return self.model.predict(state)[0]

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: Game state.
            :return int: Chosen action.
        """
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        mv = np.amax(q)  # determine max value
        actions = np.nonzero(q == mv)[0]  # get index of the action(s) with the max value
        return random.choice(actions)
