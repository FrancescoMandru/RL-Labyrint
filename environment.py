import numpy as np
import scipy.special as sp
import dill
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot as plt

class Environment:

    state = []
    goal = []
    boundary = []
    'Stay,Right,Left,Down,Up'
    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }
    # Obstacles
    walls = [[2,1],[2,2],[2,3],[2,5],[2,6],[3,5],[3,6],
             [4,1],[4,3],[4,5],[4,6],[4,7],[4,8],[5,1],[5,3],
             [5,8],[6,1],[6,3],[6,4],[6,5],[6,8],[7,8],
             [8,1],[8,2],[8,3],[8,4],[8,5],[8,6]]
    sand = [[1,7],[1,8],[2,7],[2,8],[3,7],[3,8],[7,3],[7,4],[7,5]]

    def __init__(self, x, y, initial, goal, walls=walls, sand=sand):
        # Define the grid dimension
        self.boundary = np.asarray([x, y])
        # Decide the initial position of the agent
        self.state = np.asarray(initial)
        # Decide the position of the goal
        self.goal = goal
        self.walls = walls
        self.sand = sand

    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action, obstacles=False):
        reward = 0
        # Agent makes a move
        movement = self.action_map[action]
        # Temporary state of the agent
        temp_state = list(self.state + np.asarray(movement))
        #print('Temporary state after action: ', temp_state)
        # Actions, positive and negative rewards
        if((temp_state in self.sand) and obstacles):
            reward = -0.3
            #print('Sand!')
        elif((self.state == self.goal).all()):
            reward = 1
            #print('Goal reached!')
        elif(self.check_boundaries(self.state + np.asarray(movement))):
            reward = -0.6
            #print('Out of the grid!')
        elif((temp_state in self.walls) and obstacles):
            reward = -0.1
            #print('Wall hit!')
        else:
            reward = 0
            #print('Nothing found')
        if(reward == 0.3 or reward == -0.3 or reward == 0 or reward == 1):
            #print('Reward achieved')
            self.state = self.state + np.asarray(movement)
            #print('State after move: ', self.state)
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        # Mi dice di quante dimensioni sono uscito
        out = len([num for num in state if num < 0])
        # Mi dice di quanto sono uscito dalla griglia
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0
