###Using various ways of reinforcement learning to escape from a maze.

The maze is the environment in which an agent is placed somewhere. The agents goal is to reach the exit. 
In order to get there the agent constantly chooses actions (move left/right/up/down). With every step
the agent moves to the next state of the game, and a penalty or reward is incurred. These penalties and
rewards are input for training the model which the agent uses to choose the best next action. The models
are based on Q-learning.

This project demonstrates four different models to move through a maze. The main file is *game.py*. 
Here the environment and agent are defined. Near the end of the file a maze is defined, a model is
trained and then used to play a number of games starting from different positions in the maze. 
During the play the agents moves are plotted.

Package *models* contains the following models:
1. *QRandom* is the simplest model and just selects the next move randomly. It does not learn
2. *QTable* uses a table which maps states to Q values. Q represents the quality of each action. These Q's are constantly refined during training.
3. *QNetwork* is a simple neural network which learns the relation between a state and the corresponding Q's by playing lots of games.
4. *QReplayNetwork* is a network which learns by replaying previous games. After learning it saves the model so this can be loaded for a next game (which does not need to learn). 

Requires matplotlib, numpy, keras and tensorflow.

![](https://github.com/erikdelange/Reinforcement-Learning-Maze/blob/master/ui.png)