import numpy as np
import random

class QLearnMDP:
    """Q Learning Algorithm with epsilon greedy policy

    Arguments:
        env: Environment 
        alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        gamma: Discount Rate --> How much importance we want to give to future rewards
        epsilon: Probability of selecting random action instead of the 'optimal' action
        episodes: No. of episodes to train 

    Returns:
        Q-learning Trained policy

    """
    
    """Training the agent"""
    def __init__(self, env, alpha, gamma, epsilon, episodes):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.epRewards = np.zeros(episodes)

    def train(self):

        for episode in range(self.episodes):
            state = self.env.reset()[0]
            done = False
            reward=0
            total_reward = 0
            
            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()     # Explore state space
                else:
                    action = np.argmax(self.q_table[state])     # Exploit learned values
                
                next_state, reward, done, truncated, info = self.env.step(action)   # invoke Gym
                next_max = np.max(self.q_table[next_state])

                old_value = self.q_table[state, action]
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

                self.q_table[state, action] = new_value         # Update Q table

                total_reward += reward 
                state = next_state
                
            self.epRewards[episode] = total_reward

            if episode % 100 == 0:
                print("Episode: {:d} / {:d}, Reward: {:.1f}".format(episode, self.episodes, total_reward))

    def evaluate(self, epochs):
        total_epochs, total_penalties = 0, 0
        for epoch in range(epochs):
            state = self.env.reset()[0]
            eps, penalities, reward = 0, 0, 0
            done = False
            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, truncated, info = self.env.step(action)
                if reward == -10:
                    penalities += 1
                eps += 1
            total_penalties += penalities
            total_epochs += eps
            print("Results at {:d} episode:".format(epoch))
            print("Average timesteps per episode: {:.1f}".format(total_epochs / (epoch+1)))
            print("Average penalties per episode: {:.1f}".format(total_penalties / (epoch+1)))
        
        # output results
        print("Results after {:d} episodes:".format(epochs))
        print("Average timesteps per episode: {:.1f}".format(total_epochs / epochs))
        print("Average penalties per episode: {:.1f}".format(total_penalties / epochs))