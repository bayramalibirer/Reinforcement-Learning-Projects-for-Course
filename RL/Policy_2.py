import torch
import gym
import matplotlib.pyplot as plt

# CartPole-v1 çevresini oluştur
env = gym.make('CartPole-v1')

# Durum uzayının boyutu ve eylem uzayındaki eylemlerin sayısı
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

# Bir bölüm boyunca ajanı çalıştıran fonksiyon
def run_episode(env, weight):
    state_tuple = env.reset()
    state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
    grads = []
    total_reward = 0
    is_done = False
    while not is_done:
        state = torch.from_numpy(state).float()
        z = torch.matmul(state, weight)
        probs = torch.nn.Softmax(dim=0)(z)
        action = int(torch.bernoulli(probs[1]).numpy().item())
        d_softmax = torch.diag(probs) - probs.view(-1, 1) * probs
        d_log = d_softmax[action] / probs[action]
        grad = state.view(-1, 1) * d_log
        grads.append(grad)
        state, reward, truncated, is_done, info = env.step(action)
        total_reward += reward
    return total_reward, grads

# Eğitim parametreleri
n_episode = 1000
learning_rate = 0.001

total_rewards = []

# Ağırlıkları rastgele başlat
weight = torch.rand(n_state, n_action)

# Politika gradyanı metodu ile ajanı eğit
for episode in range(n_episode):
    total_reward, gradients = run_episode(env, weight)
    for i in range(int(total_reward)):
        weight += learning_rate * gradients[i]
    total_rewards.append(total_reward)

# Eğitim sonuçlarını görselleştir
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.show()

# Eğitilmiş ajanın performansını değerlendir
n_episode_eval = 100
total_rewards_eval = []
for episode in range(n_episode_eval):
    total_reward, _ = run_episode(env, weight)
    total_rewards_eval.append(total_reward)

# Değerlendirme sonuçlarını yazdır
print('Average total reward over {} episodes: {}'.format(n_episode_eval, sum(total_rewards_eval) / n_episode_eval))
