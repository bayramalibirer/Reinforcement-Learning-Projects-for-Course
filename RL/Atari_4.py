import gym
from ale_py import ALEInterface
from ale_py.roms import Breakout

ale = ALEInterface()
env=gym.make("SpaceInvaders-v4")

env.reset()

env.render()

action = env.action_space.sample()
new_state, reward, is_done, info = env.step(action)
print(is_done)
print(info)
env.render()


is_done = False
while not is_done:
    action = env.action_space.sample()
    new_state, reward, is_done, info = env.step(action)
    print(info)
    env.render()

print(info)

print(env.action_space)

print(new_state.shape)
