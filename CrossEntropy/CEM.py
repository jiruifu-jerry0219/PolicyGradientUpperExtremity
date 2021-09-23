from osim.env import Arm2DEnv
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from CEM_Agent import Agent

# Set up CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test the environment
env = Arm2DEnv(visualize=False)
env.seed(101)

print("======== Information of Observation Space ========")
print('observation space:', env.observation_space)
print('observation shape:', env.observation_space.shape[0])

print("======== Information of Action Space ========")
print('action space:', env.action_space)
print('action shape:', env.action_space.shape[0])
print(" - low:", env.action_space.low)
print(" - high:", env.action_space.high)

np.random.seed(101)



agent = Agent(env).to(device, 64)

n_iterations = 500
max_t = 1000
gamma = 1.0
print_every = 10
pop_size = 50
elite_frac = 0.2
sigma = 0.5

n_elite=int(pop_size*elite_frac)

scores_deque = deque(maxlen=100)
scores = []
best_weight = sigma*np.random.randn(agent.get_weights_dim())

for i_iteration in range(1, n_iterations+1):
    weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
    rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])

    elite_idxs = rewards.argsort()[-n_elite:]
    elite_weights = [weights_pop[i] for i in elite_idxs]
    best_weight = np.array(elite_weights).mean(axis=0)

    reward = agent.evaluate(best_weight, gamma=1.0)
    scores_deque.append(reward)
    scores.append(reward)

    torch.save(agent.state_dict(), 'checkpoint.pth')

    if i_iteration % print_every == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

    if np.mean(scores_deque)>=90.0:
        print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
        break

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
fig.savefig('CEM_RESULT.png')
