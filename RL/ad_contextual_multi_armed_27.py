# Solving internet advertising problems with contextual bandits
# A multi-armed bandit is a single machine with multiple arms while contextual bandits are aset of such machines (bandits)
# Each machine in contextual bandits is a state that has multiple arms


import torch
from multi_armed_bandit import BanditEnv

# Define the payout probabilities and reawards for the 2 three-armed bandits
bandit_payout_machines = [
    [0.01, 0.015, 0.03],
    [0.025, 0.01, 0.015]
]
bandit_reward_machines = [
    [1, 1, 1],
    [1, 1, 1]
]
# The true CTR of ad 0 is 1%, of ad 1 is 1.5%, and of ad 2 is 3% for the first state and [2.5%, 1% and 1.5% ] for the second state
n_machine = len(bandit_payout_machines)

bandit_env_machines = [BanditEnv(bandit_payout, bandit_reward)
                       for bandit_payout, bandit_reward in
                       zip(bandit_payout_machines, bandit_reward_machines)]

n_episode = 100000
n_action = len(bandit_payout_machines[0])
action_count = torch.zeros(n_machine, n_action)
action_total_reward = torch.zeros(n_machine, n_action)
action_avg_reward = [[[] for action in range(n_action)] for _ in range(n_machine)]

# Define the UCB policy function, which computes the best arm based on UCB formula:

def upper_confidence_bound(Q, action_count, t):
    ucb = torch.sqrt((2 * torch.log(torch.tensor(float(t)))) / action_count) + Q
    return torch.argmax(ucb)

# Initialzie the Q function which is the average reward obtained with individual arms for individual states:

Q_machines = torch.empty(n_machine, n_action)

# We will update the Q-function over time
# Now we will run 100000 episodes with the UCB policy. For each episode, we also update the statistics of each arm in each state.

for episode in range(n_episode):
    state = torch.randint(0, n_machine, (1,)).item()

    action = upper_confidence_bound(Q_machines[state], action_count[state], episode)
    reward = bandit_env_machines[state].step(action)
    action_count[state][action] += 1
    action_total_reward[state][action] += reward
    Q_machines[state][action] = action_total_reward[state][action] / action_count[state][action]

    for a in range(n_action):
        if action_count[state][a]:
            action_avg_reward[state][a].append(action_total_reward[state][a] / action_count[state][a])
        else:
            action_avg_reward[state][a].append(0)


import matplotlib.pyplot as plt
for state in range(n_machine):
    for action in range(n_action):
        plt.plot(action_avg_reward[state][action])
    plt.legend(['Arm {}'.format(action) for action in range(n_action)])
    plt.xscale('log')
    plt.title('Average reward over time for state {}'.format(state))
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.show()

