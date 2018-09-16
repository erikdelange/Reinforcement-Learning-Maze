import logging

import matplotlib.pyplot as plt
import numpy as np

CELL_EMPTY = 0  # indicates empty cell where the agent can move to
CELL_OCCUPIED = 1  # indicates cell contains a wall and cannot be entered
CELL_CURRENT = 2  # indicates current cell of the agent

# all actions the agent can choose, plus a dictionary for textual representation
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
    """ A maze with walls. An agent is placed at the start cell and moves through the maze to get to the exit cell.

        The maze is the environment in which an agent is placed at start_cell. The agent chooses actions
        (move left/right/up/down) in order to reach the exit_cell. Every action results in a reward/penalty which is
        accumulated during the game. Every move gives a small penalty (-0.04), returning to a cell the agent visited
        earlier a bigger penalty(-0.25) and running into a wall a large penalty (-0.75). A large reward (+1)is
        collected when reaching the exit. The game always reaches a terminal state; the agent wins or looses.
        Reaching the exit means winning. If the penalties the agent has collected during play exceed a threshold
        the agent is assumed to wander around cluelessly and looses.

        A note in cell coordinates:
        The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze.
        This way of storing coordinates is in line with what the plot() function expects as inputs. The maze itself
        is stored as a 2D array so cells are accessed via [row, col]. To convert a (col, row) tuple
        to (row, col) use: (col, row)[::-1]
    """

    def __init__(self, maze, start_cell=(0, 0), exit_cell=None):
        """ Create a new maze with a specific start- and exit-cell.

            :param numpy.array maze: 2D Array containing empty cells (=0) and cells occupied with walls (=1).
            :param tuple start_cell: Starting cell for the agent in the maze (optional, else upper left).
            :param tuple exit_cell: Exit cell which the agent has to reach (optional, else lower right).
        """
        self.maze = maze
        self.display = False  # draw grid and moves or not
        self.__minimum_reward = -0.5 * self.maze.size  # stop game if accumulated reward is below this threshold

        self.actions = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN]

        nrows, ncols = self.maze.shape
        exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell

        self.__exit_cell = exit_cell
        self.__previous_cell = self.current_cell = start_cell
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == CELL_EMPTY]

        if exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(exit_cell))
        if self.maze[exit_cell[::-1]] == CELL_OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(exit_cell))

        self.empty.remove(exit_cell)
        self.reset(start_cell)

    def reset(self, start_cell=(0, 0)):
        """ Reset the maze to its initial state and place the agent at start_cell.

            :param tuple start_cell: Cell where the agent starts its journey through the maze (optional, else upper left).
            :return: New state after reset.
        """
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == CELL_OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.__previous_cell = self.current_cell = start_cell
        self.__total_reward = 0.0
        self.__visited = set()

        if self.display:
            # draw the initial maze
            nrows, ncols = self.maze.shape
            plt.clf()
            plt.xticks(np.arange(0.5, nrows, step=1), [])
            plt.yticks(np.arange(0.5, ncols, step=1), [])
            plt.grid(True)
            plt.plot(*self.current_cell, "rs", markersize=25)  # start is a big red square
            plt.plot(*self.__exit_cell, "gs", markersize=25)  # exit is a big green square
            plt.imshow(self.maze, cmap="binary")
            plt.pause(0.01)

        return self.__observe()

    def __draw(self):
        """ Draw a line from the agents previous to its current cell. """
        plt.plot(*zip(*[self.__previous_cell, self.current_cell]), "bo-")  # previous cells are blue dots
        plt.plot(*self.current_cell, "ro")  # current cell is a red dot
        plt.pause(0.01)

    def step(self, action):
        """ Move the agent according to action and return the new state, reward and game status.

            :param int action: The agent will move in this direction.
            :return: state, reward, status
        """
        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        return state, reward, status

    def __execute(self, action):
        """ Execute action and collect the reward/penalty.

            :param int action: The agent will move in this direction.
            :return float: Reward/penalty after the action is done.
        """
        possible_actions = self.__possible_actions(self.current_cell)

        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move, force end of game
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

            self.__previous_cell = self.current_cell
            self.current_cell = (col, row)

            if self.current_cell == self.__exit_cell:
                reward = 1.0  # maximum reward for reaching the exit cell
            elif self.current_cell in self.__visited:
                reward = -0.25  # penalty for returning to a cell which was visited earlier
            else:
                reward = -0.04  # penalty for a move which did not result in finding the exit cell

            self.__visited.add(self.current_cell)
        else:
            reward = -0.75  # penalty for trying to enter an occupied cell (= a wall) or moving out of the maze

        return reward

    def __possible_actions(self, cell=None):
        """ Create a list with possible actions taking into account the maze's edges and walls.

            :param tuple cell: Location of the agent (optional, else current cell).
            :return list: All possible actions.
        """
        if cell is None:
            col, row = self.current_cell
        else:
            col, row = cell

        possible_actions = self.actions.copy()  # initially allow all

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

    def __status(self):
        """ Determine the game status.

            :return str: Current game status (win/lose/playing).
        """
        if self.current_cell == self.__exit_cell:
            return "win"

        if self.__total_reward < self.__minimum_reward:  # force end after to much loss
            return "lose"

        return "playing"

    def __observe(self):
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

        state = self.__observe()

        while True:
            action = model.predict(state=state)
            state, reward, status = self.step(action)
            if self.display:
                logging.info("action: {:10s} | reward: {: .2f} | status: {}".format(actions[action], reward, status))
                self.__draw()
            if status in ("win", "lose"):
                return status

    def win_all(self, model):
        """ Check if the model wins from all possible starting cells. """
        for cell in self.empty:
            if self.play(model, cell) == "lose":
                return False
        else:
            return True


if __name__ == "__main__":
    from models import RandomModel, QTableModel, QNetworkModel, QReplayNetworkModel

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s: %(asctime)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    maze = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0]
    ])  # 0 = free, 1 = occupied

    game = Maze(maze)

    # change index to choose the model to use
    index = 1
    test = ["random", "qtable", "qnetwork", "qreplay", "load previous qreplay"][index]

    if test == "random":
        model = RandomModel(game)
        model.train()
    elif test == "qtable":
        model = QTableModel(game)
        model.train(episodes=500)
    elif test == "qnetwork":
        model = QNetworkModel(game)
        model.train(episodes=500)
    elif test == "qreplay":
        model = QReplayNetworkModel(game)
        model.train(episodes=maze.size * 10, max_memory=maze.size * 8, modelname="test")
    else:
        model = QReplayNetworkModel(game, load=True)

    game.display = True
    game.play(model, start_cell=(0, 0))
    game.play(model, start_cell=(2, 5))
    game.play(model, start_cell=(4, 1))

    plt.show()  # must be placed here else the image disappears immediately at the end of the program
