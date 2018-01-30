import gym
from torch.distributions import Categorical
from utils import *
import os
from models import *
from config import *
import numpy as np

# 3 is right 4 is left 5 is shooting + right is 6 shooting + left 7
def main():
    env = gym.make('BeamRider-v0')
    env.reset()
    observation, reward, done, info = env.step(2)  # Get Reward for action
    while True:
        env.render()

        action = np.random.randint(1, 9)

        observation, reward, done, info = env.step(action)  # Get Reward for action
        if done:
            observation = env.reset()


if __name__ == '__main__':
    main()
