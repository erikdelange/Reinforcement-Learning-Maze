### Using various ways of reinforcement learning to escape from a maze.

The environment for this problem is a maze with walls and an exit. An agent is initially placed somewhere in the maze. The agents goal is to reach the exit. To get there the agent moves through the maze in a succession of steps. At every step the agent decides which action to take (move left/right/up/down). For this purpose the agent uses a prediction model. With every step the agent incurs a penalty or reward. These penalties and rewards are input for training the model which the agent uses to choose the best next action. 

The models are based on Q-learning. During training the Q-learning algorithm updates the value for each action in a certain state. The highest value indicates the most preferable action. Updating the values is based on the reward or penalty incurred after the action was taken. With Q-learning a model learns at every step it takes, not only when the exit is reached. However learning does speed up once the exit has been reached for the first time. 

This project demonstrates four different models to move through a maze. The main file is *game.py*. Here the environment and agent are defined in class Maze. Near the end of the file an actual maze is defined as an np.array, a model is trained and then used to play a number of games starting from different starting positions in the maze. When playing the agents moves can be plotted.

Package *models* contains the following models:
1. *RandomModel* is the simplest model and just selects the next move randomly. It does not learn
2. *QTableModel* uses a table which maps states to Q values. Q represents the quality of each action. These Q's are constantly refined during training.
3. *QNetworkModel* is a simple neural network which learns the relation between a state and the corresponding Q's by playing lots of games.
4. *QReplayNetworkModel* is a network which learns by replaying previous games. After learning it saves the model to disk so this can be loaded later for a next game (which then does not need to learn).

![](https://github.com/erikdelange/Reinforcement-Learning-Maze/blob/master/maze.png)

Requires matplotlib, numpy, keras and tensorflow.
