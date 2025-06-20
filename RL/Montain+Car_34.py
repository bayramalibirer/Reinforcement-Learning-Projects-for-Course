import gym

import torch
env = gym.make('MountainCarContinuous-v0')

print(env.action_space.low[0])

print(env.action_space.high[0])

env.reset()

is_done = False
while not is_done:
    random_action = torch.rand(1) * 2 - 1
    next_state, reward, is_done, info = env.step(random_action)
    print(next_state, reward, is_done)
    env.render()

env.render()
