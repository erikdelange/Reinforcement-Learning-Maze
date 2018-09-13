import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, model_from_json

CELL_EMPTY = 0  # indicates empty cell where the agent can move to
CELL_OCCUPIED = 1  # indicates cell contains a wall and cannot be entered
CELL_CURRENT = 2  # indicates current cell of the agent

# all moves the agent can make, plus a dictionary for textual representation
MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_UP = 2
MOVE_DOWN = 3

actions = {
    MOVE_LEFT: "move left",
    MOVE_RIGHT: "move right",
    MOVE_UP: "move up",
    MOVE_DOWN: "move down"
}


class Maze:
    """ A maze with walls. An agent is placed at the start cell and needs to move through the maze to reach the exit cell.

    An agent begins its journey at start_cell. The agent can execute moves (up/down/left/right) in the maze in order
    to reach the exit_cell. Every move results in a reward/penalty which is accumulated during the game. Every move
    gives a small penalty, returning to a cell the agent visited earlier a bigger penalty and running into a wall a
    large penalty. The reward is only collected when reaching the exit. The game always has a terminal state; in
    the end you win or lose. You lose if you have collected a large number of penalties; then the agent is
    assumed to wander around cluelessly.

    Cell coordinates:
    The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze. This
    way of storing coordinates is in line with what the plot() function expects as inputs.
    The maze itself is stored as a 2D array so cells are accessed via [row, col]. To convert a (col, row) tuple to
    (row, col) use: (col, row)[::-1] -> (row, col)
    """

    def __init__(self, maze, start_cell=(0, 0), exit_cell=None):
        """ Create a new maze with a specific start- and exit cell.

        :param numpy.array maze: 2D Array containing empty cells (=0) and cells occupied with walls (=1).
        :param tuple start_cell: Starting cell for the agent in the maze (optional, else upper left).
        :param tuple exit_cell: Exit cell which the agent has to reach (optional, else lower right).
        """
        self.maze = maze
        self.display = False  # draw grid and moves or not
        self.minimum_reward = -0.5 * self.maze.size  # stop game if accumulated reward is below this number

        nrows, ncols = self.maze.shape
        exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell

        self.exit_cell = exit_cell
        self.previous_cell = self.current_cell = start_cell
        self.cells = [(c, r) for c in range(ncols) for r in range(nrows)]
        self.empty = [(c, r) for c in range(ncols) for r in range(nrows) if maze[r, c] == CELL_EMPTY]

        if exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(exit_cell))
        if self.maze[exit_cell[::-1]] == CELL_OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(exit_cell))

        self.empty.remove(exit_cell)
        self.reset(start_cell)

    def reset(self, start_cell=(0, 0)):
        """ Reset the maze to its initial state and place the agent at start_cell.

        :param tuple start_cell: Cell where the agent will start its journey through the maze.
        """
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == CELL_OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.previous_cell = self.current_cell = start_cell
        self.total_reward = 0.0
        self.visited = set()

        # if display has been enabled then draw the initial maze
        if self.display:
            nrows, ncols = self.maze.shape
            plt.clf()
            plt.xticks(np.arange(0.5, nrows, step=1), [])
            plt.yticks(np.arange(0.5, ncols, step=1), [])
            plt.grid(True)
            plt.plot(*self.current_cell, "rs", markersize=25)  # start is a big red square
            plt.plot(*self.exit_cell, "gs", markersize=25)  # exit is a big green square
            plt.imshow(self.maze, cmap="binary")
            plt.pause(0.05)

    def show(self):
        """ Enable display of the maze and all moves. """
        self.display = True

    def hide(self):
        """ Hide the maze. """
        self.display = False

    def draw(self):
        """ Draw a line from the agents previous to its current cell. """
        plt.plot(*zip(*[self.previous_cell, self.current_cell]), "bo-")  # previous cells are blue dots
        plt.plot(*self.current_cell, "ro")  # current cell is a red dot
        plt.pause(0.05)

    def move(self, action):
        """ Move the agent according to action and return the new state, reward and game status.

        :param int action: The direction of the agents move.
        :return: state, reward, status
        """
        reward = self.update_state(action)
        self.total_reward += reward
        status = self.status()
        state = self.observe()
        return state, reward, status

    def update_state(self, action):
        """ Execute action and collect the reward/penalty.

        :param int action: The direction in which the agent will move.
        :return float: Reward/penalty after the action has been executed.
        """
        possible_actions = self.possible_actions(self.current_cell)

        if not possible_actions:
            reward = self.minimum_reward - 1  # cannot move any more, force end of game
        elif action in possible_actions:
            col, row = self.current_cell
            if action == MOVE_LEFT:
                col -= 1
            elif action == MOVE_UP:
                row -= 1
            if action == MOVE_RIGHT:
                col += 1
            elif action == MOVE_DOWN:
                row += 1

            self.previous_cell = self.current_cell
            self.current_cell = (col, row)

            if self.current_cell == self.exit_cell:
                reward = 1.0  # maximum reward for reaching the exit cell
            elif self.current_cell in self.visited:
                reward = -0.25  # penalty for returning to a cell which was visited earlier
            else:
                reward = -0.04  # penalty for a move which did not result in finding the exit cell

            self.visited.add(self.current_cell)
        else:
            reward = -0.75  # penalty for trying to enter a occupied cell (= a wall)

        return reward

    def possible_actions(self, cell=None):
        """ Create a list with possible actions (taking into account the maze's edges and walls).

        :param tuple cell: Location of the agent (optional, else current cell).
        :return list: All possible actions.
        """
        if cell is None:
            col, row = self.current_cell
        else:
            col, row = cell

        possible_actions = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN]  # initially allow all

        # now restrict the initial list by removing impossible actions
        nrows, ncols = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_DOWN)

        if col == 0 or (col > 0 and self.maze[row, col - 1] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_RIGHT)

        return possible_actions

    def status(self):
        """ Determine the game status.

        :return str: Current game status (win/lose/playing).
        """
        if self.current_cell == self.exit_cell:
            return "win"
        if self.total_reward < self.minimum_reward:  # force end after to much loss
            return "lose"

        return "playing"

    def observe(self):
        """ Create a [1][Z] copy of the maze (Z = total cell count in the maze), including the agents current location.

        :return numpy.array [1][size]: Maze content as an array of 1*total_cells_in_array.
        """
        state = np.copy(self.maze)
        col, row = self.current_cell
        state[row, col] = CELL_CURRENT  # indicate the agents current location
        return state.reshape((1, -1))

    def play(self, model, start_cell=(0, 0)):
        """ Play a single game, choosing the next move based a prediction from the model.

        :param model: The prediction model to use.
        :param tuple start_cell: Agents initial cell (optional, else upper left).
        :return str: "win" or "lose"
        """
        self.reset(start_cell)

        state = self.observe()

        while True:
            action = model.predict(state)
            state, reward, status = self.move(action)
            if self.display:
                logging.info("action: {:10s} | reward: {: .2f} | status: {}".format(actions[action], reward, status))
                self.draw()
            if status in ("win", "lose"):
                return status


class ExperienceReplay:
    """ Store game transitions (from state s to s') and update Q's. Replay these to train the model. """

    def __init__(self, model, max_memory=1000, discount=0.9):
        """ Create the replay memory.

        :param model: A Keras NN model.
        :param int max_memory: How many consecutive game transitions to store.
        :param float discount: How important are future rewards (0 = not at all, 1 = only)
        """
        self.model = model
        self.discount = discount
        self.memory = list()
        self.max_memory = max_memory

    def remember(self, transition):
        """ Add a game transition at the end of the memory list.

        :param list transition: [state, move, reward, next_state, status]
        """
        self.memory.append(transition)
        if len(self.memory) > self.max_memory:
            del self.memory[0]  # forget the oldest memories

    def predict(self, state):
        """ Predict the Q vector belonging to this state.

        :param np.array state: A game state.
        :return: np.array with Q's per action.
        """
        return self.model.predict(state)[0]  # prediction is a [1][num_actions] array with Q's

    def get_samples(self, sample_size=10):
        """ Retrieve a number of random observed game states and the corresponding Q target vectors.

        :param int sample_size: The number of states to return
        :return: input and target vectors (as np.array)
        """
        mem_size = len(self.memory)  # how many episodes are currently stored
        sample_size = min(mem_size, sample_size)  # cannot take more samples then available in memory
        state_size = self.memory[0][0].size  # number of cells in maze
        num_actions = self.model.output_shape[-1]  # number of actions based in output layer

        states = np.zeros((sample_size, state_size), dtype=int)
        targets = np.zeros((sample_size, num_actions), dtype=float)

        # update the Q's using the Bellman equation
        for i, idx in enumerate(np.random.choice(range(mem_size), sample_size, replace=False)):
            state, move, reward, next_state, status = self.memory[idx]

            states[i] = state
            targets[i] = self.predict(state)

            if status == "win":
                targets[i, move] = reward
            else:
                targets[i, move] = reward + self.discount * np.max(self.predict(next_state))

        return states, targets


class DeepQNetwork:
    def __init__(self, game, modelname="model", load=False):
        self.game = game

        if load is False:
            self.model = Sequential()
            self.model.add(Dense(game.maze.size, input_shape=(game.maze.size,), activation="relu"))
            self.model.add(Dense(game.maze.size, activation="relu"))
            self.model.add(Dense(len(actions)))
        else:
            self.load(modelname)

        self.model.compile(optimizer="adam", loss="mse")

    def save(self, filename):
        """ Save a model plus weights. """
        with open(filename + ".json", "w") as outfile:
            outfile.write(self.model.to_json())
        self.model.save_weights(filename + ".h5", overwrite=True)

    def load(self, filename):
        """ Load a model plus weights. """
        with open(filename + ".json", "r") as infile:
            self.model = model_from_json(infile.read())
        self.model.load_weights(filename + ".h5")

    def train(self, **kwargs):
        """ Tune the Q network.

        Takes a sample from previous action and fit the model on this sample.
        """
        epsilon = 0.2  # exploration vs exploitation (0 = only exploit, 1 = only explore)
        episodes = kwargs.get("episodes", 10000)
        sample_size = kwargs.get("sample_size", 10)
        load_weights = kwargs.get("load_weights", False)
        modelname = kwargs.get("modelname", "model")

        if load_weights:
            self.model.load_weights(modelname + ".h5")

        experience = ExperienceReplay(self.model, discount=0.5)

        wins = 0

        for episode in range(1, episodes):
            loss = 0.0

            start_cell = random.choice(self.game.empty)
            self.game.reset(start_cell)

            state = self.game.observe()

            while True:
                possible_actions = self.game.possible_actions()
                if not possible_actions:
                    status = "blocked"
                    break
                if np.random.random() < epsilon:
                    action = random.choice(possible_actions)
                else:
                    action = np.argmax(experience.predict(state))

                next_state, reward, status = self.game.move(action)

                experience.remember([state, action, reward, next_state, status])

                if status in ("win", "lose"):
                    if status == "win":
                        wins += 1
                    break

                inputs, targets = experience.get_samples(sample_size=sample_size)

                # loss = self.model.train_on_batch(inputs, targets)
                h = self.model.fit(inputs,
                                   targets,
                                   epochs=8,
                                   batch_size=16,
                                   verbose=0)
                loss = self.model.evaluate(inputs, targets, verbose=0)

                state = next_state

            logging.info("episode: {:05d}/{:05d} | loss: {:.4f} | total wins: {:04d} ({:.2f})"
                         .format(episode, episodes, loss, wins, wins / episodes))

            # check if with current model we win from all starting cells
            if episode > self.game.maze.size and ((episode % 25) == 0):
                for cell in self.game.empty:
                    if (self.game.play(self, start_cell=cell)) == "lose":
                        break
                else:
                    # won in all cases
                    logging.info("Won from all start cells, exit learning")
                    break

        self.save(modelname)  # Save trained models weights and architecture

    def predict(self, state):
        """ Choose the next move based in the highest Q from the Q network.

        :param np.array state: A game state.
        :return str: The action.
        """
        q = self.model.predict(state)
        action = int(np.argmax(q[0]))
        logging.debug("q = {} | max = {}".format(q, actions[action]))
        return action


class QNetwork:
    def __init__(self, game):
        """ A simple neural network for Q-learning.

        The network learns the Q values for each action in each state. In the model the state is represented as a
        vector which maps to Q values.

        :param Maze game: Maze game object.
        """
        self.game = game

        self.model = Sequential()
        self.model.add(Dense(game.maze.size, input_shape=(game.maze.size,), activation="relu"))
        self.model.add(Dense(game.maze.size, activation="relu"))
        self.model.add(Dense(len(actions)))
        self.model.compile(optimizer="adam", loss="mse")

    def train(self, **kwargs):
        """ Tune the Q network by playing a number of games (called episodes).

        Dependent on epsilon, take a random action or base the action on the current Q network. Update the
        Q network after every action using the Bellman equation.
        """
        # hyperparameters
        epsilon = 0.1  # exploration vs exploitation (0 = only exploit, 1 = only explore)
        discount = 0.9  # (gamma) importance of future rewards (0 = not at all, 1 = only)
        episodes = kwargs.get("episodes", 1000)  # number of training games to play

        wins = 0

        for episode in range(1, episodes):
            start_cell = random.choice(self.game.empty)
            self.game.reset(start_cell)

            state = self.game.observe()

            while True:
                possible_actions = self.game.possible_actions()
                if not possible_actions:
                    status = "blocked"
                    break
                if np.random.random() < epsilon:
                    action = random.choice(possible_actions)
                else:
                    action = np.argmax(self.model.predict(state))

                next_state, reward, status = self.game.move(action)

                if status in ("win", "lose"):
                    target = reward  # no discount needed if a terminal state was reached.
                else:
                    target = reward + discount * np.max(self.model.predict(next_state))

                target_vector = self.model.predict(state)
                target_vector[0][action] = target  # update Q value for this action

                self.model.fit(state, target_vector, epochs=1, verbose=0)

                if status in ("win", "lose"):
                    if status == "win":
                        wins += 1
                    break

                state = next_state

            logging.info("episode: {:05d}/{:05d} | status: {:4s} | total wins: {:4d} ({:.2f})"
                         .format(episode, episodes, status, wins, wins / episode))

    def predict(self, state):
        """ Choose the next move based in the highest Q from the Q network.

        :param np.array state: A game state.
        :return str: The action.
        """
        q = self.model.predict(state)
        action = int(np.argmax(q[0]))
        logging.debug("q = {} | max = {}".format(q, actions[action]))
        return action


class QTable:
    """ Reinforcement learning via Q-table.

        For every state (= maze layout with the agents current location ) the Q for each of the actions is stored.
        By playing games enough games (= training), and for every move updating the Q's according to the Bellman
        equation, good quality Q's are determined. If the Q's are good enough can be tested by playing a game.

        Note that this implementation scales badly if the size of the maze increases. """

    def __init__(self, game):
        """ Create a Q-table for all possible states. The q's for each action are initially set to 0.

        State is the maze layout + the location of agent in the maze.
        Todo: try to replace this by just the agents current cell as this is the only thing which changes

        :param Maze game: Maze game object.
        """
        self.game = game
        self.qtable = dict()

        for cell in game.cells:
            state = np.copy(self.game.maze)
            col, row = cell
            state[row, col] = CELL_CURRENT
            state = tuple(
                state.flatten())  # convert [1][Z] array to a tuple of array[Z] so it can be used as dictionary key
            self.qtable[state] = [0, 0, 0, 0]  # 4 possible actions, initially all equally good/bad

    def train(self, **kwargs):
        """ Tune the Q-table by playing a number of games (called episodes).

        Take a random action, of base the action on the current Q table. Update the Q table after each action
        using the Bellman equation.
        """
        # hyperparameters
        epsilon = 0.1  # exploration vs exploitation (0 = only exploit, 1 = only explore)
        discount = 0.9  # (gamma) importance of future rewards (0 = not at all, 1 = only)
        learning_rate = 0.3  # (alpha) speed of learning (0 = no learning, only exploit, 1 = only use most recent information)
        episodes = kwargs.get("episodes", 1000)  # number of training games to play

        wins = 0

        for episode in range(1, episodes):
            start_cell = random.choice(self.game.empty)
            self.game.reset(start_cell)

            state = self.game.observe()
            state = tuple(state.flatten())

            while True:
                possible_actions = self.game.possible_actions()
                if not possible_actions:
                    status = "blocked"
                    break
                if np.random.random() < epsilon:
                    action = random.choice(possible_actions)
                else:
                    action = np.nanargmax(self.qtable[state])  # note: this argmax version ignores nan's

                next_state, reward, status = self.game.move(action)
                next_state = tuple(next_state.flatten())

                self.qtable[state][action] += learning_rate * (
                        reward + discount * max(self.qtable[next_state]) - self.qtable[state][action])

                if status in ("win", "lose"):
                    if status == "win":
                        wins += 1
                    break

                state = next_state

            logging.info("episode: {:04d}/{:05d} | status: {:4s} | total wins: {:4d} ({:.2f})"
                         .format(episode, episodes, status, wins, wins / episode))

        # replace any initial zero still left for a nan (not-a-number)
        for key in self.qtable:
            self.qtable[key] = [np.nan if q == 0 else q for q in self.qtable[key]]

    def predict(self, state):
        """ Choose the next move based in the highest Q from the Q-table.

        :param np.array state: A game state.
        :return str: The action.
        """
        state = tuple(state.flatten())
        q = self.qtable[state]
        action = int(np.nanargmax(q))  # action is the index of the highest Q value
        logging.debug("q = {} | max = {}".format(q, actions[action]))
        return action


class Random:
    """ Choose random moves when playing a game. """

    def __init__(self, game):
        self.game = game

    def predict(self, state):
        """ Choose the next move randomly.

        :param np.array state: A game state.
        :return int: The action or None if no action is possible.
        """
        possible_actions = self.game.possible_actions()
        if not possible_actions:
            return None
        else:
            return random.choice(possible_actions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s: %(asctime)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    maze = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0]
    ])  # 0 = free, 1 = occupied

    if 0:
        game = Maze(maze)
        model = Random(game)
        game.show()
        game.play(model, start_cell=(0, 0))

    if 1:
        game = Maze(maze)
        model = QTable(game)
        model.train(episodes=500)
        game.show()
        game.play(model, start_cell=(0, 0))
        game.play(model, start_cell=(0, 4))
        game.play(model, start_cell=(3, 7))

    if 0:
        game = Maze(maze)
        model = QNetwork(game)
        model.train(episodes=100)
        game.show()
        game.play(model, start_cell=(0, 0))
        game.play(model, start_cell=(0, 4))
        game.play(model, start_cell=(3, 7))

    if 0:
        game = Maze(maze)
        model = DeepQNetwork(game)
        model.train(episodes=maze.size * 10, max_memory=maze.size * 8, modelname="test")
        game.show()
        game.play(model, start_cell=(0, 0))
        game.play(model, start_cell=(0, 4))
        game.play(model, start_cell=(3, 7))

    if 0:
        game = Maze(maze)
        model = DeepQNetwork(game, load=True)
        game.show()
        game.play(model, start_cell=(0, 0))
        game.play(model, start_cell=(0, 4))
        game.play(model, start_cell=(3, 7))

    plt.show()  # must be here else the image disappears immediately at the end of the program
