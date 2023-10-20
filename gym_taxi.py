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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# Set up Seaborn style
sns.set(style="darkgrid")
Efont_prop = FontProperties(fname="C:\Windows\Fonts\ARLRDBD.TTF")
label_prop = FontProperties(family='serif', size=7, weight='normal')
legend_font = FontProperties(family='serif', size=7, weight='normal')

if __name__ == '__main__':

    opts = BaseOptions().parse()         # set project's options

    # Set OpenAI Gym environment
    env = gym.make('Taxi-v3', render_mode="rgb_array")

    gamma_delta = 0.01
    aver_rewards = np.zeros(len(np.arange(opts.lb_gamma, opts.ub_gamma + gamma_delta, gamma_delta)))
    random_aver_rewards = np.zeros(aver_rewards.shape)
    cum_rewards = np.zeros(aver_rewards.shape)
    random_cum_rewards = np.zeros(aver_rewards.shape)
    for t, gamma in enumerate(np.arange(opts.lb_gamma, opts.ub_gamma + gamma_delta, gamma_delta)):
        # Init env and value iteration process
        VIMDP = ValueMDP(env, opts, gamma)
    
        # Apply the random policy
        VIMDP.env.reset(seed=t+101)
        VIMDP.ApplyRandomPolicy(steps=10)

        # Value Iteration in MDP
        observation = VIMDP.env.reset(seed=t+101)
        VIMDP.IterateValueFunction()
        
        # Apply the optimal policy
        VIMDP.ApplyOptimalPolicy(observation[0], steps=20)

        # Save reward results
        aver_rewards[t] = VIMDP.aver_reward
        random_aver_rewards[t] = VIMDP.random_aver_reward
        cum_rewards[t] = VIMDP.cum_reward
        random_cum_rewards[t] = VIMDP.random_cum_reward
        print("discount factor: %f" % gamma)
        print("Applying the random policy, accumulated reward: %.5f, average reward: %.5f" % (random_cum_rewards[t], random_aver_rewards[t]))
        print("Applying the optimal policy, accumulated reward: %.5f, average reward: %.5f" % (cum_rewards[t], aver_rewards[t]))

    # plot the rewards
    xdata = np.arange(opts.lb_gamma, opts.ub_gamma + gamma_delta, gamma_delta)
    plt.subplot(211)
    plt.plot(xdata, random_aver_rewards, 'b-', label='random policy')
    plt.plot(xdata, aver_rewards, 'g-', label='optimal policy')
    plt.ylabel('Average Rewards', fontproperties=Efont_prop, fontsize=9)
    plt.yticks(fontproperties=label_prop, fontsize=7)
    plt.xticks(fontproperties=label_prop, fontsize=7)
    plt.legend(loc='lower right', fontsize=7, prop=legend_font)

    plt.subplot(212)
    plt.plot(xdata, random_cum_rewards, 'b--', label='random policy')
    plt.plot(xdata, cum_rewards, 'g--', label='optimal policy')
    plt.xlabel('Discount Factor', fontproperties=Efont_prop, fontsize=9) 
    plt.ylabel('Cumulative Rewards', fontproperties=Efont_prop, fontsize=9)
    plt.yticks(fontproperties=label_prop, fontsize=7)
    plt.xticks(fontproperties=label_prop, fontsize=7)
    plt.legend(loc='lower right', fontsize=7, prop=legend_font)

    plt.savefig("Rewards.png", dpi=400)
    env.close()