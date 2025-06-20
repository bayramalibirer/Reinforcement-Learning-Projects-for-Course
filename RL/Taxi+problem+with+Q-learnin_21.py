import torch
import gym
from collections import defaultdict

env = gym.make('Taxi-v3')


def gen_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


def q_learning(env, gamma, n_episode, alpha):
    """
    Obtain the optimal policy with off-policy Q-learning method
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        while not is_done:

            action = epsilon_greedy_policy(state, Q)
            next_state, reward, is_done, info = env.step(action)
            td_delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
            Q[state][action] += alpha * td_delta

            length_episode[episode] += 1
            total_reward_episode[episode] += reward

            if is_done:
                break
            state = next_state

    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy



n_episode = 1000

n_state = env.observation_space.n
print(n_state)

n_action = env.action_space.n
print(n_action)

env.reset()
env.render()

print(env.step(3))

print(env.step(3))

print(env.step(3))

print(env.step(1))


print(env.step(1))

print(env.step(4))

env.render()


print(env.step(0))

print(env.step(0))

print(env.step(0))


print(env.step(0))

print(env.step(5))


env.render()

length_episode = [0] * n_episode

total_reward_episode = [0] * n_episode


gamma = 1
alpha = 0.4
epsilon = 0.1

epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)

optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha)

import matplotlib.pyplot as plt
plt.plot(length_episode)
plt.title('Episode length over time')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.show()


plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()


















