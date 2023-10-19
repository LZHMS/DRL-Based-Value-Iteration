"""
-------------------------------------------------------
Project: Solving as MDP using Value Iteration Algorithm
Author: Zhihao Li
Date: October 19, 2023
Research Content: Deep Reinforcement Learning
-------------------------------------------------------
"""

from options import BaseOptions
from value_iteration import ValueMDP
import gym                 # openAi gym
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    opts = BaseOptions().parse()         # set project's options

    # Set OpenAI Gym environment
    env = gym.make('Taxi-v3')

    for t_rounds in range(opts.n_rounds):
        # Init env and value iteration process
        observation = env.reset()
        VIMDP = ValueMDP(env, opts)

        # Value Iteration in MDP
        VIMDP.IterateValueFunction()

        # Apply policy
        VIMDP.ApplyPolicy(observation, steps=1000)

        if t_rounds % opts.print_interval == 0 and t_rounds > 0:
            print(VIMDP.cum_reward * 1.0 / (t_rounds + 1))

    env.close()