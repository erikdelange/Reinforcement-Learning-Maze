import matplotlib.pyplot as plt

from environment import Maze
from models import *

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
    game.render("moves")
    game.reset()

if 0:  # play using random model
    model = RandomModel(game)
    model.train()

if 0:  # train using tabular Q-learning
    model = QTableModel(game, name="QTableModel")
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200)

if 0:  # train using tabular Q-learning and an eligibility trace (aka TD-lamba)
    model = QTableTraceModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200)

if 0:  # train using tabular SARSA learning
    model = SarsaTableModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200)

if 1:  # train using tabular SARSA learning and an eligibility trace
    game.render("training")  # shows all moves and the q table; nice but slow.
    model = SarsaTableTraceModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200)

if 0:  # train using a neural network with experience replay (also saves the resulting model)
    model = QReplayNetworkModel(game)
    h, w, _, _ = model.train(discount=0.80, exploration_rate=0.10, episodes=maze.size * 10, max_memory=maze.size * 4)

try:
    h  # force a NameError exception if h does not exist (and thus don't try to show win rate and cumulative reward)
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.canvas.set_window_title(model.name)
    ax1.plot(*zip(*w))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    ax2.plot(h)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except NameError:
    pass

if 0:  # load a previously trained model
    model = QReplayNetworkModel(game, load=True)

if 0:  # compare learning speed (cumulative rewards and win rate) of several models in a diagram
    rhist = list()
    whist = list()
    names = list()

    models = [0, 1, 2, 3, 4]

    for model_id in models:
        logging.disable(logging.WARNING)
        if model_id == 0:
            model = QTableModel(game, name="QTableModel")
        elif model_id == 1:
            model = SarsaTableModel(game, name="SarsaTableModel")
        elif model_id == 2:
            model = QTableTraceModel(game, name="QTableTraceModel")
        elif model_id == 3:
            model = SarsaTableTraceModel(game, name="SarsaTableTraceModel")
        elif model_id == 4:
            model = QReplayNetworkModel(game, name="QReplayNetworkModel")

        r, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, exploration_decay=0.999, learning_rate=0.10,
                                 episodes=300)
        rhist.append(r)
        whist.append(w)
        names.append(model.name)

    f, (rhist_ax, whist_ax) = plt.subplots(2, len(models), sharex="row", sharey="row", tight_layout=True)

    for i in range(len(rhist)):
        rhist_ax[i].set_title(names[i])
        rhist_ax[i].set_ylabel("cumulative reward")
        rhist_ax[i].plot(rhist[i])

    for i in range(len(whist)):
        whist_ax[i].set_xlabel("episode")
        whist_ax[i].set_ylabel("win rate")
        whist_ax[i].plot(*zip(*(whist[i])))

    plt.show()

if 0:  # run a number of training episodes and plot the training time and episodes needed in histograms (time consuming)
    runs = 10

    epi = list()
    nme = list()
    sec = list()

    models = [0, 1, 2, 3, 4]

    for model_id in models:
        episodes = list()
        seconds = list()

        logging.disable(logging.WARNING)
        for r in range(runs):
            if model_id == 0:
                model = QTableModel(game, name="QTableModel")
            elif model_id == 1:
                model = SarsaTableModel(game, name="SarsaTableModel")
            elif model_id == 2:
                model = QTableTraceModel(game, name="QTableTraceModel")
            elif model_id == 3:
                model = SarsaTableTraceModel(game, name="SarsaTableTraceModel")
            elif model_id == 4:
                model = QReplayNetworkModel(game, name="QReplayNetworkModel")

            _, _, e, s = model.train(stop_at_convergence=True, discount=0.90, exploration_rate=0.10,
                                     exploration_decay=0.999, learning_rate=0.10, episodes=1000)

            print(e, s)

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

    plt.show()

game.render("moves")
game.play(model, start_cell=(0, 0))
# game.play(model, start_cell=(2, 5))
# game.play(model, start_cell=(4, 1))

plt.show()  # must be placed here else the image disappears immediately at the end of the program
