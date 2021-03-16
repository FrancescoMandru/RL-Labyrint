import numpy as np
import scipy.special as sp
import dill
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot as plt
import agent, environment


def printConfiguration( config ):
    print('\n'.join('- {}: {}'.format(*k) for k in list(zip(config.keys(), config.values())) ))
    print('\n')

mode = 'agent'

if mode=='paramsearch':

    episodes = 5000                      # number of training episodes
    x = 10                                # horizontal size of the box
    y = 10                                # vertical size of the box
    goal = [5, 7]                         # objective point
    softmax = True                       # set to true to use Softmax policy
    sarsa = True                          # set to true to use the Sarsa algorithm

    params = {'episode_length':[100,50],
            'discount':[0.7, 0.6, 0.5, 0.4],
            'alpha_val':[0.9,0.7,0.5,0.6,0.4],
            'eps_val':[0.7,0.6,0.5,0.4,0.3],
            'Softmax':[True,False]
            }

    grid = ParameterGrid( params )

    best_reward = 0
    best_conf = 0

    for i,prm in enumerate(grid):
        print('\n\n --- Config. number %u of %u' %(i,len(grid)))
        printConfiguration( prm )
        # TODO alpha and epsilon profile
        alpha = np.ones(episodes) * prm['alpha_val']
        epsilon = np.linspace(prm['eps_val'], 0.001, episodes)

        # initialize the agent
        learner = agent.Agent((x * y), 5, prm['discount'], max_reward=1, softmax=prm['Softmax'], sarsa=sarsa)

        episode_length = prm['episode_length']
        # perform the training
        reward_idx = []
        for index in range(0, episodes):
            # start from a random state
            initial = [np.random.randint(0, x), np.random.randint(0, y)]
            # initialize environment
            state = initial
            env = environment.Environment(x, y, state, goal)
            reward = 0
            # run episode
            for step in range(0, episode_length):
                # find state index
                state_index = state[0] * y + state[1]
                # choose an action
                action = learner.select_action(state_index, epsilon[index])
                # the agent moves in the environment
                result = env.move(action)
                # Q-learning update
                next_index = result[0][0] * y + result[0][1]
                learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
                # update state and reward
                reward += result[1]
                state = result[0]
            reward /= episode_length
            reward_idx.append(reward)

        print('Final reward: ', reward)
        print('Average reward: {0:.2f} ' .format(np.mean(reward_idx)))
        if(np.mean(reward_idx) > best_reward):
            best_reward = np.mean(reward_idx)
            best_conf = i


    print('BEST MODEL')
    print('----\n')
    printConfiguration( list(grid)[best_conf] )

if mode=='agent':

    episodes = 10000                      # number of training episodes
    x = 10                                # horizontal size of the box
    y = 10                                # vertical size of the box
    goal = [[5, 7]]                   # objective point


    ##### PARAMETERS ######
    softmax = False
    sarsa = True
    obstacles = True
    alpha_val = 0.5
    epsilon_val = 0.3
    discount_val = 0.7
    episode_length = 50
    ##### ---------- #####

    alpha = np.linspace(alpha_val, 0.0001, episodes)
    # Standard epsilon-greedy
    #epsilon = [e for e in np.linspace(epsilon_val, 0.0001, episodes)]
    eps_prct = int(episodes*0.7)
    # Modified version of epsilon greedy strategy
    epsilon = [e for e in np.linspace(epsilon_val, 0.0001, episodes-eps_prct)]
    for i in range(episodes-(episodes-eps_prct)):
      epsilon.append(0)

    print('\n\n PARAMETERS SELECTED')
    print('\n ----- Episodes: ', episodes)
    print('\n ----- alpha: ', alpha_val)
    print('\n ----- epsilon: ', epsilon_val)
    print('\n ----- discount_val: ', discount_val)
    print('\n ----- SARSA: ', sarsa)
    print('\n ----- Softmax: ', softmax)
    print('\n ----- Obstacles: ', obstacles)
    print('\n ----- Episode length: ', episode_length)
    print('\n --------------------',)
    # initialize the agent
    learner = agent.Agent((x * y), 5, discount_val, max_reward=1, softmax=softmax, sarsa=sarsa)


    # perform the training
    for index in range(0, episodes):
        #print('New Episode')
        # start from a random state
        initial = [np.random.randint(0, x), np.random.randint(0, y)]
        # initialize environment
        state = initial
        env = environment.Environment(x, y, state, goal)
        # Check if the agent start inside a wall, which is not possible!
        while(((list(initial) in env.walls) and obstacles) or (list(initial) in goal)):

          initial = [np.random.randint(0, x), np.random.randint(0, y)]

        state = initial
        #print('Initial state: ', state)
        env = environment.Environment(x, y, state, goal)
        reward = 0
        # run episode
        new_episode_length = 1
        #print('\n\n New Episode')
        for step in range(1, episode_length+1):
            # find state index
            state_index = state[0] * y + state[1]
            # choose an action
            action = learner.select_action(state_index, epsilon[index])
            #print('Action: ', action)
            # the agent moves in the environment
            result = env.move(action, obstacles)
            #print('REWARD from state {} at step {} is {}'.format(result[0],step,result[1]))
            # Q-learning update
            #print('State: ', result[0])
            #print('Reward: ', result[1])
            next_index = result[0][0] * y + result[0][1]
            learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
            # update state and reward
            reward += result[1]
            state = result[0]
            #print('Goal reached at step {} for the episode {}'.format(step,index))

        reward /= episode_length
        #print('Average Reward: ', reward)
        #periodically save the agent
        if ((index + 1) % 100 == 0):
          with open('agent.obj', 'wb') as agent_file:
            dill.dump(learner, agent_file)
            print('Episode ', index + 1, ': the agent has obtained an average reward of {0:.2f} '.format( reward ),
                  ' starting from position ', initial)
