import logging
import random
from datetime import datetime

import numpy as np

np.random.seed(1)

import tensorflow

tensorflow.compat.v1.random.set_random_seed(2)

from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from environment import Status
from models import AbstractModel


class ExperienceReplay:
    """ Store game transitions (from state s to s' via action a) and record the rewards. When
        a sample is requested update the Q's.
    """

    def __init__(self, model, max_memory=1000, discount=0.95):
        """
        :param model: Keras NN model.
        :param int max_memory: number of consecutive game transitions to store
        :param float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
        """
        self.model = model
        self.discount = discount
        self.memory = list()
        self.max_memory = max_memory

    def remember(self, transition):
        """ Add a game transition at the tail of the memory list.

            :param list transition: [state, move, reward, next_state, status]
        """
        self.memory.append(transition)
        if len(self.memory) > self.max_memory:
            del self.memory[0]  # forget the oldest memories

    def predict(self, state):
        """ Predict the Q vector belonging to this state.

            :param np.array state: game state
            :return np.array: array with Q's per action
        """
        return self.model.predict(state)[0]  # prediction is a [1][num_actions] array with Q's

    def get_samples(self, sample_size=10):
        """ Randomly retrieve a number of observed game states and the corresponding Q target vectors.

        :param int sample_size: number of states to return
        :return np.array: input and target vectors
        """
        mem_size = len(self.memory)  # how many episodes are currently stored
        sample_size = min(mem_size, sample_size)  # cannot take more samples than available in memory
        state_size = self.memory[0][0].size
        num_actions = self.model.output_shape[-1]  # number of actions in output layer

        states = np.zeros((sample_size, state_size), dtype=int)
        targets = np.zeros((sample_size, num_actions), dtype=float)

        # update the Q's from the sample
        for i, idx in enumerate(np.random.choice(range(mem_size), sample_size, replace=False)):
            state, move, reward, next_state, status = self.memory[idx]

            states[i] = state
            targets[i] = self.predict(state)

            if status == "win":
                targets[i, move] = reward  # no discount needed if a terminal state was reached.
            else:
                targets[i, move] = reward + self.discount * np.max(self.predict(next_state))

        return states, targets


class QReplayNetworkModel(AbstractModel):
    """ Prediction model which uses Q-learning and a neural network which replays past moves.

        The network learns by replaying a batch of training moves. The training algorithm ensures that
        the game is started from every possible cell. Training ends after a fixed number of games, or
        earlier if a stopping criterion is reached (here: a 100% win rate).
    """
    default_check_convergence_every = 5  # by default check for convergence every # episodes

    def __init__(self, game, **kwargs):
        """ Create a new prediction model for 'game'.

        :param class Maze game: maze game object
        :param kwargs: model dependent init parameters
        """
        super().__init__(game, name="QReplayNetworkModel", **kwargs)

        if kwargs.get("load", False) is False:
            self.model = Sequential()
            self.model.add(Dense(game.maze.size, input_shape=(2,), activation="relu"))
            self.model.add(Dense(game.maze.size, activation="relu"))
            self.model.add(Dense(len(game.actions)))
        else:
            self.load(self.name)

        self.model.compile(optimizer="adam", loss="mse")

    def save(self, filename):
        with open(filename + ".json", "w") as outfile:
            outfile.write(self.model.to_json())
        self.model.save_weights(filename + ".h5", overwrite=True)

    def load(self, filename):
        with open(filename + ".json", "r") as infile:
            self.model = model_from_json(infile.read())
        self.model.load_weights(filename + ".h5")

    def train(self, stop_at_convergence=False, **kwargs):
        """ Train the model.

            :param stop_at_convergence: stop training as soon as convergence is reached

            Hyperparameters:
            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
            :keyword int episodes: number of training games to play
            :keyword int sample_size: number of samples to replay for training
            :return int, datetime: number of training episodes, total time spent
        """
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        episodes = max(kwargs.get("episodes", 1000), 1)
        sample_size = kwargs.get("sample_size", 32)
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)

        experience = ExperienceReplay(self.model, discount=discount)

        # variables for reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()  # starting cells not yet used for training
        start_time = datetime.now()

        # training starts here
        for episode in range(1, episodes + 1):
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)

            loss = 0.0

            while True:
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    # q = experience.predict(state)
                    # action = random.choice(np.nonzero(q == np.max(q))[0])
                    action = self.predict(state)

                next_state, reward, status = self.environment.step(action)

                cumulative_reward += reward

                experience.remember([state, action, reward, next_state, status])

                if status in (Status.WIN, Status.LOSE):  # terminal state reached, stop episode
                    break

                inputs, targets = experience.get_samples(sample_size=sample_size)

                self.model.fit(inputs,
                               targets,
                               epochs=4,
                               batch_size=16,
                               verbose=0)
                loss += self.model.evaluate(inputs, targets, verbose=0)

                state = next_state

                self.environment.render_q(self)

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | loss: {:.4f} | e: {:.5f}"
                         .format(episode, episodes, status.name, loss, exploration_rate))

            if episode % check_convergence_every == 0:
                # check if the current model does win from all starting cells
                # only possible if there is a finite number of starting states
                w_all, win_rate = self.environment.check_win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            exploration_rate *= exploration_decay  # explore less as training progresses

        self.save(self.name)  # Save trained models weights and architecture

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == tuple:
            state = np.array(state, ndmin=2)

        return self.model.predict(state)[0]

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: game state
            :return int: selected action
        """
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)
