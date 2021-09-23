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
env = Arm2DEnv(visualize=True)
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

# load the weights from file
# load the weights from file
agent.load_state_dict(torch.load('checkpoint.pth'))

state = env.reset()
n = 0
i = 0
while True:
    i += 1
    state = torch.from_numpy(state).float().to(device)
    with torch.no_grad():
        action = agent(state)
    env.render()
    n += 1
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if i >= 1000:
        break
    if done:
        print('Finished the task in: {} steps'.format(n))
        n = 0
        state = env.reset()

env.close()
