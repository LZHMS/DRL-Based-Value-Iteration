# Deep Reinforcement Learning Based on Value Iteration
## Theories
### Markov Decision Process
Generally, we notes a MDP model as $(S, A, T_a, R_a, \gamma)$. Its transition function is $T_a(s,s')=\Pr(s_{t+1}|s_t=s, a_t=a)$, reward function is $R_a(s,s')$. And actions choosing satisfies a specific distribution.
The cotinuous decisions are noted as trace $\tau$, formally in formula:
<center>$\tau=${$s_t, a_t, r_t, s_{t+1}, \cdots, a_{t+n}, r_{t+n}, s_{t+n+1}$}</center>

And in many situations, we very care about the expected reward of a specific trace because that will support us to choose the optimal action currently. So we use the method like weighted time series to calculate cumulative reward:
$$
R(\tau_t) = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots=r_t+\sum_{i=1}^\infty \gamma^ir_{t+i}
$$

After we got the return value of traces, we can just calculate the value of a state to form our policy.
$$
V^{\pi}(s)=E_{\tau\sim p(\tau_t)}[\sum_{i=0}^\infty \gamma^ir_{t+i}|s_t=s]
$$
However, although we can get the value function to form optimal policy, we cann't still calculate the values of all states. So we need Bellmax Equation to solve the problem.
### Bellman Equation
$$
V^{\pi}(s)=\sum_{a\in A}\pi(a|s)[\sum_{s'\in S}T_a(s,s')[R_a(s,s')+\gamma V^{\pi}(s')]
$$

For a specific state $s$, when choosing some action, we will get a stochastic new state which satisfies some distribution. *Bellman Equation* tells us to calculate the expected average value of these possible new states' return. And in detail, the return of each state have two parts: the immediate reward $R_a(s,s')$ and the future reward $\gamma V^{\pi}(s')$. That inspires us that we can calculate the value of states recursively.

### Value Iteration
Value Iteration is a method to calculate *Bellman Equation* by traversing the state and action space. Firstly, it stores a value table of all states. And in traversing process, it will calculate the value of each state and update the value table by choosing the action with the highest return.

## Experiments
### Taxi Environment of OpenAI Gym
+ Taxi Enviroment
The Taxi example is an environment where taxis move up, down, left, and right, and pichup and dropoff passengers. There are four disignated locations in the Grid world indicated by R(ed), B(lue), G(reen), and Y(ellow).
+ Taxi Activities
In an episode, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, then drives to the passenger's  destination(another one of the four specified locations), and drops off the passenger.
+ States and Actions Space
    + $500=25\times5\times4$ discrete states
    With the grid size of $5 \times 5$, there are $25$ taxi positions. For the passenger, there are $5$ possible locations(including the case when the passenger is in the taxi). For the destination, there are $4$ possible locations.
    + $6$ discrete deterministic actions
    For the Taxi diver, 
        + $0$: Move south
        + $1$: Move north
        + $2$: Move east
        + $3$: Move west
        + $4$: Pick up passenger
        + $5$: Drop off passenger
+ Rewards
    + $-1$ for each action
    + $+20$ for delivering the passenger
    + $-10$ for picking up and dropping off the passenger illegally


The following pictures are taxi example demostration. The left shows taxi actions with a random policy and the right shows taxi actions with the optimal policy.
<div class="justified-gallery">
<img src="https://cdn.jsdelivr.net/gh/LZHMS/picx-images-hosting@master/DRL/randomPolicy.pnt0kxzusv4.gif" alt="randomPolicy" />
<img src="https://cdn.jsdelivr.net/gh/LZHMS/picx-images-hosting@master/DRL/optimalPolicy.2wskea2qtzi0.gif" alt="optimalPolicy" />
</div>

## Results
Now we want to check how the discount factor influences the value function from the same start state. So we choosing the discount factor ranging from $0.0$ to $1.0$ with footstep of 0.05 to measure the average rewards and cumulative rewards on random group and optimal group.

<img src="https://cdn.jsdelivr.net/gh/LZHMS/picx-images-hosting@master/DRL/Rewards.2uwj07wcwru0.webp" alt="Tuning MDP Results" width="70%"/>

|Discount Factor|Random Cum_Reward|Random_Aver_Reward|Optimal Cum_Reward|Optimal_Aver_Reward|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|0.00|-37|-3.70|-20|-2.00|
|0.05|-10|-1.00|-20|-1.00|
|0.10|-55|-5.50|10|0.91|
|0.15|-37|-3.70|11|1.10|
|0.20|-55|-5.50|-20|-1.00|
|0.25|-28|-2.80|15|2.50|
|0.30|-46|-4.60|11|1.10|
|0.35|-28|-2.80|5|0.31|
|0.40|-10|-1.00|7|0.50|
|0.45|-37|-3.70|7|0.50|
|0.50|-64|-6.40|7|0.50|
|0.55|-19|-1.90|13|1.60|
|0.60|-28|-2.80|9|0.75|
|0.65|-46|-4.60|10|0.91|
|0.70|-37|-3.70|9|0.75|
|0.75|-46|-4.60|6|0.40|
|0.80|-37|-3.70|4|0.24|
|0.85|-37|-3.70|7|0.50|
|0.90|-28|-2.80|7|0.50|
|0.95|-37|-3.70|5|0.31|
|1.00|-37|-3.70|11|1.10|

## Conclusions
From the following experimental results, we can conclude that the discount factor has a significant impact on the value function. The optimal group has a higher average and cumulative reward than the random group, and the discount factor has a lower bound $\gamma=0.4$ to get optimal policy.
In my opinion, the discount factor reflects the future reward's influence on the current state. If it is set too small, that means the most reward comes from the immediate reward which is a greedy policy with the possibility of failure. On the other hand, if set too high, we also cann't get the best action with the highest reward. So we'd better to set the discount factor to an appropriate value.


## Contributors
+ [Zhihao Li](https://lzhms.github.io/)

## References
+ [OpenAI Gym](https://www.gymlibrary.dev/)

