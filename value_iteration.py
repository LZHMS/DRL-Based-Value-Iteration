import numpy as np

"""
--------------------------------------------------------------------------------------
This section is for Value Iteration Algorithm for Taxi Gym.
Author: Zhihao Li
Date: October 19, 2023
Arguments:
    env: OpenAI env. env.P represents the transition probabilities of the environment.
        env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
    theta: Stop evaluation once value function change is less than theta for all states.
    discount_factor: Gamma discount factor.
--------------------------------------------------------------------------------------
"""

class ValueMDP:

    def __init__(self, env, opts) -> None:
        self.env = env                    # taxi gym environment
        self.gamma = opts.gamma           # discount_factor
        self.NA = opts.NA                 # Actions Space's Length
        self.NS = opts.NS                 # States Space's Length
        self.V = np.zeros(self.NS)        # Value Function
        self.end_delta = opts.end_delta   # Delta value for stopping iteration
        self.new_policy = np.zeros([self.NS, self.NA])    # the optimal policy
        self.cum_reward = 0               # apply new policy and get all rewards

    def SingleStepIteration(self, state):
        """
        Function: calculate the state value for all actions in a given state 
                  and update the value function.
        Returns:
            The estimate of actions.
        """
        action_V = np.zeros(self.NA)     # Record the value of each action
        for action in range(self.NA):
            for prob, nextState, reward, is_final in self.env.P[state][action]:
                action_V[action] += prob * (reward + self.gamma * self.V[nextState] * (not is_final))

        return action_V
    
    def IterateValueFunction(self):

        while True:
            delta = 0           # initialize the every round of delta
            
            for s in range(self.NS):
                newValue = np.max(self.SingleStepIteration(s))
                delta = max(delta, np.abs(newValue - self.V[s]))
                self.V[s] = newValue          # updates value function
            
            if delta < self.end_delta:    # the maximum delta of all states
                break
        
        # get optimal policy
        for s in range(self.NS):         # for all states, create deterministic policy
            newAction = np.argmax(self.SingleStepIteration(s))
            self.new_policy[s][newAction] = 1

    def ApplyPolicy(self, observation, steps):
        for _ in range(steps):
            action = self.new_policy[observation[0]]

            observation, reward, is_final, truncated, info = self.env.step[np.argmax(action)]
            self.cum_reward += reward
            if is_final:
                break
