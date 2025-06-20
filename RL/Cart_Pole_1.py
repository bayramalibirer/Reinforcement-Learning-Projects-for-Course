import gym
import numpy as np

env = gym.make('CartPole-v1',render_mode='human')
np.random.seed(1)
HighRevard = 0
BestWeights = None

for i in range(20):
    result = env.reset()
    observation = result[0] if isinstance(result, tuple) else result
    Weights= np.random.uniform(-1,1,4)
    SumReward = 0
    for j in range(100):
        env.render()
        action = 0 if np.matmul(Weights, observation) < 0 else 1
        observation, reward, truncated, done, info = env.step(action)
        SumReward += reward if reward is not None else 0
        print(i,j)
        print('observation:', observation)
        print('reward:', reward)
        print('done:', done)
        print('info:', info)
        print('SumReward:', SumReward)
        print('Weights:', Weights)
        print('BestWeights:', BestWeights)
        if done:
            break
    if SumReward > HighRevard:
        HighRevard = SumReward
        BestWeights = Weights

    # observation : An environment-specific object representing your observation of the environment.
    # reward : Amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
    # done : This determines whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
    # info : This shows you diagnostic information helpful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). However, official evaluations of your agent are not allowed to use this for learning.
    # the observation object contains 4 parameters: cart position, cart velocity, pole angle, pole velocity at tip
    # there are 2 possible objects: push the cart to the left(0) or to the right(1)
result = env.reset()
observation = result[0] if isinstance(result, tuple) else result
for j in range(100):
    env.render()
    action = 0 if np.matmul(BestWeights, observation) < 0 else 1
    observation, reward, truncated, done, info = env.step(action)
    print(j,action)
