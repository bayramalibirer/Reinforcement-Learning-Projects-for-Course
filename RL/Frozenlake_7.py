# S: the starting location
# G: The gole location which terminates an episode
# F: THe frozen title, which is a walkable location
# H: The hole location which terminates an episode

# four actions : moving left (0), moving down (1), moving right (2), moving up (3) and
import gym
import torch
env = gym.make("FrozenLake-v1",render_mode='rgb_array')
n_state = env.observation_space.n
print(n_state)

n_action = env.action_space.n
print(n_action)

env.reset()

env.render()

new_state,reward, truncated, is_done, info= env.step(1)
env.render()

print(new_state)

print(reward)

print(is_done)

print(info)

def run_episode(env, policy):
    result = env.reset()
    state = result[0] if isinstance(result, tuple) else result
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward,truncated, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward

n_episode = 1000

total_rewards = []
for episode in range(n_episode):
    random_policy = torch.randint(high=n_action, size=(n_state,))
    total_reward = run_episode(env, random_policy)
    total_rewards.append(total_reward)

print('Average total reward under random policy: {}'.format(sum(total_rewards) / n_episode))

while True:
    random_policy = torch.randint(high=n_action, size=(n_state,))
    total_reward = run_episode(env, random_policy)
    if total_reward == 1:
        best_policy = random_policy
        break
total_rewards = []
for episode  in range(n_episode):
    total_reward = run_episode(env, best_policy)
    total_rewards.append(total_reward)

print(best_policy)
print('Average total reward under random search policy: {}'.format(sum(total_rewards) / n_episode))


print(env.env.P[6])
# The movement list is in the following format: (transformation, probability, new state, reward receive, is done)
print(env.env.P[11])



#Aşağıdaki adımlarda 4'e 4 FrozenLake ortamını simüle edelim: 
# 1) Spor salonu kütüphanesini içe aktarıyoruz ve Frozen Lake ortamının bir örneğini oluşturuyoruz.
# 2) Ortamı sıfırlayın. 
# 3) Ortamı işleyin.
# 4) Yürünebilir olduğundan aşağı doğru bir hareket yapalım. 
# 5) Ajanın %33,33 olasılıkla durum 4'e indiğini doğrulamak için tüm geri dönen bilgileri yazdırın. 
# 6) Donmuş gölde yürümenin ne kadar zor olduğunu göstermek için rastgele bir politika uygulayın ve 1000 bölüm üzerinden ortalama toplam ödülü hesaplayın. Öncelikle, bir politika verildiğinde Frozen Lake bölümünü simüle eden ve toplam ödülü döndüren bir fonksiyon tanımlayın (bunun 0 veya 1 olduğunu biliyoruz). 
# 7) Şimdi 1000 bölümü çalıştırın; bir politika rastgele oluşturulacak ve her bölümde kullanılacaktır. 
# 8) Daha sonra rastgele bir arama politikasıyla denemeler yapıyoruz. Eğitim aşamasında rastgele bir grup politika üretiyoruz ve hedefe ulaşan ilk politikayı kaydediyoruz. 
# 9) En iyi politikaya bir göz atın. 10) Şimdi az önce seçtiğimiz politikayla 1000 bölüm çalıştırın.