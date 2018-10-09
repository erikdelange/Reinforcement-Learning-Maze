### Escape from a maze using reinforcement learning.

#####Solving an optimization problem using a MDP and TD learning. 

The environment for this problem is a maze with walls and an exit. An agent (the learner and decision maker) is placed somewhere in the maze. The agents goal is to reach the exit as quick as possible. To get there the agent moves through the maze in a succession of steps. At every step the agent decides which action to take (move left/right/up/down). For this purpose the agent is trained to learn a policy (Q), which indicates the best next step to take. With every step the agent incurs a penalty or (when finally reaching the exit) a reward. These penalties and rewards are input for training the policy. 

The policies (or models) are based on Sarsa and Q-learning. During training the learning algorithm updates the action-value function Q for each state which is visited. The highest value indicates the most preferable action. Updating the values is based on the reward or penalty incurred after the action was taken. With TD-learning a model learns at every step it takes, not only when the exit is reached. However learning does speed up once the exit has been reached for the first time. 

This project demonstrates different models to move through a maze. Class Maze in file *maze.py* in package *environment* defines the environment including the rules of the game. In file *main.py* an example of a maze is defined as an np.array. By changing *if 0* into *if 1* a certain model is trained and then used to play a number of games from different starting positions in the maze. When playing the agents moves can be plotted if Maze.display is set to True.

Package *models* contains the following models:
1. *RandomModel* is the simplest model and just selects the next move randomly. It does not learn at all. Your are lucky if you get to the exit using this model.
2. *QTableModel* uses a table which maps state plus action to a Q value. Q represents the quality of each action. These Q's are constantly refined during training. This is a fast way to learn a policy.
3. *SarsaTableModel* uses a similar setup as the previous model, but takes less risks during learning.
3. *QTableTraceModel* is an extension on the QTableModel. It speeds up learning by keeping track of the previous states-actions pairs, and updates these Q's as well although with a decaying rate. This model is trained the fastest. It uses a slightly different way to store Q's.
4. *QNetworkModel* is a simple neural network which learns the relation between a state and the corresponding Q's by playing lots of games. It is significantly slower then all other models. For the limited number of states which the Maze has this is an overkill, it is more appropriate for large state spaces.
5. *QReplayNetworkModel* is a network which learns by replaying previous games. It is the slowest of all models, but requires less training episodes then the QNetworkModel. As an extra after learning it saves the model to disk so this can be loaded later for a next game. This is typically how you would use a neural network in a real world situation where training is separated from use. 

The table below gives an impression of the relative performance of each of these models:

| Model | Trained | Average no of episodes | Average time per episode |
| --- | --- | --- | --- | 
| QTableModel | 50 times | 160.2 | 0:00:00.512832 |
| QTableTraceModel | 50 times | 111.6 | 0:00:00.505072 |
| QNetworkModel | 50 times | 257.2 | 0:02:11.391048 |
| QReplayNetworkModel | 50 times | 59.4 | 0:06:55.190230 |

![](https://github.com/erikdelange/Reinforcement-Learning-Maze/blob/master/maze.png)

Requires matplotlib, numpy, keras and tensorflow.
