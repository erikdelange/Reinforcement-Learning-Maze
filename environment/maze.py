import logging

import matplotlib.pyplot as plt
import numpy as np

CELL_EMPTY = 0  # indicates empty cell where the agent can move to
CELL_OCCUPIED = 1  # indicates cell which contains a wall and cannot be entered
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
    """ A maze with walls. An agent is placed at the start cell and must find the exit cell by moving through the maze.

        The layout of the maze and the rules how to move through it are called the environment. An agent is placed
        at start_cell. The agent chooses actions (move left/right/up/down) in order to reach the exit_cell. Every
        action results in a reward or penalty which are accumulated during the game. Every move gives a small
        penalty (-0.04), returning to a cell the agent visited earlier a bigger penalty(-0.25) and running into
        a wall a large penalty (-0.75). The reward (+1) is collected when the agent reaches the exit. The
        game always reaches a terminal state; the agent either wins or looses. Obviously reaching the exit means
        winning, but if the penalties the agent is collecting during play exceed a certain threshold the agent is
        assumed to wander around cluelessly and looses.

        A note on cell coordinates:
        The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze.
        This way of storing coordinates is in line with what matplotlibs plot() function expects as inputs. The maze
        itself is stored as a 2D numpy array so cells are accessed via [row, col]. To convert a (col, row) tuple
        to (row, col) use: (col, row)[::-1]
    """

    def __init__(self, maze, start_cell=(0, 0), exit_cell=None):
        """ Create a new maze with a specific start- and exit-cell.

            :param numpy.array maze: 2D Array containing empty cells (=0) and cells occupied with walls (=1).
            :param tuple start_cell: Starting cell for the agent in the maze (optional, else upper left).
            :param tuple exit_cell: Exit cell which the agent has to reach (optional, else lower right).
        """
        self.maze = maze
        self.__minimum_reward = -0.5 * self.maze.size  # stop game if accumulated reward is below this threshold

        self.actions = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN]

        nrows, ncols = self.maze.shape
        exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell

        self.__exit_cell = exit_cell
        self.__previous_cell = self.__current_cell = start_cell
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == CELL_EMPTY]
        self.empty.remove(exit_cell)

        if exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(exit_cell))
        if self.maze[exit_cell[::-1]] == CELL_OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(exit_cell))

        self.__render = "nothing"
        self.__ax1 = None  # axes for rendering the moves
        self.__ax2 = None  # axes for rendering the best action per cell

        self.reset(start_cell)

    def reset(self, start_cell=(0, 0)):
        """ Reset the maze to its initial state and place the agent at start_cell.

            :param tuple start_cell: Here the agent starts its journey through the maze (optional, else upper left).
            :return: New state after reset.
        """
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == CELL_OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # accumulated reward
        self.__visited = set()  # a set() only stores unique values

        if self.__render in ("training", "moves"):
            # render the maze
            nrows, ncols = self.maze.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__current_cell, "rs", markersize=25)  # start is a big red square
            self.__ax1.plot(*self.__exit_cell, "gs", markersize=25)  # exit is a big green square
            self.__ax1.imshow(self.maze, cmap="binary")
            # plt.pause(0.001)  # replaced by the two lines below
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe()

    def __draw(self):
        """ Draw a line from the agents previous to its current cell. """
        self.__ax1.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # previous cells are blue dots
        self.__ax1.plot(*self.__current_cell, "ro")  # current cell is a red dot
        # plt.pause(0.001)  # replaced by the two lines below
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()

    def render(self, content="nothing"):
        """ Define what will be rendered during play and/or training.

            :param str content: "nothing", "training" (moves and q), "moves"
        """
        if content not in ("nothing", "training", "moves"):
            raise ValueError("unexpected content: {}".format(content))

        self.__render = content
        if content == "nothing":
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if content == "training":
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Best move")
                self.__ax2.set_axis_off()
                self.render_q(None)
        if content in ("moves", "training"):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Maze")

        plt.show(block=False)

    def step(self, action):
        """ Move the agent according to 'action' and return the new state, reward and game status.

            :param int action: The agent will move in this direction.
            :return: state, reward, status
        """
        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(actions[action], reward, status))
        return state, reward, status

    def __execute(self, action):
        """ Execute action and collect the reward or penalty.

            :param int action: The agent will move in this direction.
            :return float: Reward or penalty after the action is done.
        """
        possible_actions = self.__possible_actions(self.__current_cell)

        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move anywhere, force end of game
        elif action in possible_actions:
            col, row = self.__current_cell
            if action == MOVE_LEFT:
                col -= 1
            elif action == MOVE_UP:
                row -= 1
            if action == MOVE_RIGHT:
                col += 1
            elif action == MOVE_DOWN:
                row += 1

            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)

            if self.__render != "nothing":
                self.__draw()

            if self.__current_cell == self.__exit_cell:
                reward = 2.0  # maximum reward for reaching the exit cell
            elif self.__current_cell in self.__visited:
                reward = -0.25  # penalty for returning to a cell which was visited earlier
            else:
                reward = -0.04  # penalty for a move which did not result in finding the exit cell

            self.__visited.add(self.__current_cell)
        else:
            reward = -0.75  # penalty for trying to enter an occupied cell (= a wall) or moving out of the maze

        return reward

    def __possible_actions(self, cell=None):
        """ Create a list with possible actions, avoiding the maze's edges and walls.

            :param tuple cell: Location of the agent (optional, else current cell).
            :return list: All possible actions.
        """
        if cell is None:
            col, row = self.__current_cell
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
        if self.__current_cell == self.__exit_cell:
            return "win"

        if self.__total_reward < self.__minimum_reward:  # force end of game after to much loss
            return "lose"

        return "playing"

    def __observe(self):
        """ Return the state of the maze - in this example the agents current location.

            :return numpy.array [1][2]: Agents current location.
        """
        return np.array([[*self.__current_cell]])

    def play(self, model, start_cell=(0, 0)):
        """ Play a single game, choosing the next move based a prediction from 'model'.

            :param class AbstractModel model: The prediction model to use.
            :param tuple start_cell: Agents initial cell (optional, else upper left).
            :return str: "win" or "lose"
        """
        self.reset(start_cell)

        state = self.__observe()

        while True:
            action = model.predict(state=state)
            state, reward, status = self.step(action)
            if status in ("win", "lose"):
                return status

    def win_all(self, model):
        """ Check if the model wins from all possible starting cells. """
        previous = self.__render
        self.__render = "nothing"  # avoid rendering anything during execution of win_all()

        win = 0
        lose = 0

        for cell in self.empty:
            if self.play(model, cell) == "win":
                win += 1
            else:
                lose += 1

        logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win, lose, win / (win + lose)))

        self.__render = previous
        result = True if lose == 0 else False
        return result, win / (win + lose)

    def render_q(self, model):
        """ Render the recommended action for each cell. """
        if self.__render != "training":
            return

        nrows, ncols = self.maze.shape

        self.__ax2.clear()
        self.__ax2.set_xticks(np.arange(0.5, nrows, step=1))
        self.__ax2.set_xticklabels([])
        self.__ax2.set_yticks(np.arange(0.5, ncols, step=1))
        self.__ax2.set_yticklabels([])
        self.__ax2.grid(True)
        self.__ax2.plot(*self.__exit_cell, "gs", markersize=25)  # exit is a big green square

        for cell in self.empty:
            state = cell
            q = model.q(state) if model is not None else [0, 0, 0, 0]
            mv = np.amax(q)  # determine max value
            a = np.nonzero(q == mv)[0]

            for action in a:
                dx = 0
                dy = 0
                if action == 0:  # left
                    dx = -0.2
                if action == 1:  # right
                    dx = +0.2
                if action == 2:  # up
                    dy = -0.2
                if action == 3:  # down
                    dy = 0.2

                self.__ax2.arrow(*cell, dx, dy, head_width=0.2, head_length=0.1)

        self.__ax2.imshow(self.maze, cmap="binary")
        self.__ax2.get_figure().canvas.draw()
        # plt.pause(0.001)
