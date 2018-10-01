import logging

import matplotlib.pyplot as plt
import numpy as np

from environment import Maze
from models import QTableModel, QNetworkModel, QReplayNetworkModel, QTableTraceModel

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S")

maze = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0]
])  # 0 = free, 1 = occupied

game = Maze(maze)

if 0:  # only show the maze
    game.display = True
    game.reset()

if 0:  # play using random model
    model = RandomModel(game)
    model.train()

if 0:  # train using Q table
    model = QTableModel(game)
    model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=10000)

if 1:  # train using a Q table and eligibility trace
    # game.display = True  # uncomment for direct view of progress (nice but slow)
    model = QTableTraceModel(game)
    model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=10000)

if 0:  # train using a simple neural network
    model = QNetworkModel(game)
    model.train(discount=0.90, exploration_rate=0.10, episodes=10000)

if 0:  # train using a neural network with experience replay, saves the results
    model = QReplayNetworkModel(game)
    model.train(discount=0.90, exploration_rate=0.10, episodes=maze.size * 100, max_memory=maze.size * 8)

if 0:  # load a previously trained model
    model = QReplayNetworkModel(game, load=True)

if 0:  # log the average training time per model (takes a few hours)
    """ Run a number of training episodes and plot the results in histograms. Time consuming. """
    runs = 50

    epi = list()
    nme = list()
    sec = list()

    models = [0, 1, 2, 3]

    for model_id in models:
        episodes = list()
        seconds = list()

        logging.disable(logging.WARNING)
        for r in range(runs):
            if model_id == 0:
                model = QTableModel(game, name="QTableModel")
            elif model_id == 1:
                model = QTableTraceModel(game, name="QTableTraceModel")
            elif model_id == 2:
                model = QNetworkModel(game, name="QNetworkModel")
            elif model_id == 3:
                model = QReplayNetworkModel(game, name="QReplayNetworkModel")

            _, e, s = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=10000)
            episodes.append(e)
            seconds.append(s.seconds)

        logging.disable(logging.NOTSET)
        logging.info("model: {} | trained {} times | average no of episodes: {}| average training time {}"
                     .format(model.name, runs, np.average(episodes), np.sum(seconds) / len(seconds)))

        epi.append(episodes)
        sec.append(seconds)
        nme.append(model.name)

    f, (epi_ax, sec_ax) = plt.subplots(2, len(models), sharex="row", sharey="row", tight_layout=True)

    for i in range(len(epi)):
        epi_ax[i].set_title(nme[i])
        epi_ax[i].set_xlabel("training episodes")
        epi_ax[i].hist(epi[i], edgecolor="black")

    for i in range(len(sec)):
        sec_ax[i].set_xlabel("seconds per episode")
        sec_ax[i].hist(sec[i], edgecolor="black")

game.display = True
# game.play(model, start_cell=(0, 0))
# game.play(model, start_cell=(2, 5))
# game.play(model, start_cell=(4, 1))

plt.show()  # must be placed here else the image disappears immediately at the end of the program
