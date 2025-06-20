import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Ortamı oluştur
env = gym.make("ALE/SpaceInvaders-v5")

# Aksiyon ve gözlem alanlarını tanımla
action_space_size = env.action_space.n
observation_space_shape = env.observation_space.shape

# Modeli oluştur
model = tf.keras.Sequential([
    Dense(128, activation="relu", input_shape=observation_space_shape),
    Dense(64, activation="relu"),
    Dense(action_space_size, activation="linear")
])

# Modeli derle
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Eğitim döngüsü
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy politika ile aksiyon seç
        epsilon = 0.1
        if np.random.rand() < epsilon:
            action = np.random.randint(action_space_size)
        else:
            action = np.argmax(model.predict(np.expand_dims(state, axis=0)))

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Deneyim hafızasına kaydet
        # (state, action, reward, next_state, done)

        # Modeli güncelle
        # (DQN veya başka bir RL algoritması kullanarak)

        state = next_state

    print(f"Episode {episode + 1} - Toplam Ödül: {total_reward}")

# Eğitilmiş modeli kaydet
model.save("space_invaders_model.h5")
