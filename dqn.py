import keras
import random
import numpy as np
from keras import layers
from collections import deque

GAMMA = 0.99  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 50
BUFFER_SIZE = 100
TARGET_UPDATE_FREQ = 10  # Frequency of updating the target network
LR = 0.001

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

class DQNAgent:
  def __init__(self, state_size, action_size, gamma=GAMMA, lr=LR, batch_size=BATCH_SIZE, memory_size=BUFFER_SIZE, target_update_freq=TARGET_UPDATE_FREQ):
    self.state_size = state_size
    self.action_size = action_size
    self.gamma = gamma # Discount factor
    self.lr = lr # Learning rate
    self.batch_size = batch_size
    self.target_update_freq = target_update_freq
    self.memory_size = memory_size
    self.n_games = 0

    # Replay memories
    self.memory = ReplayBuffer(self.memory_size)

    # Deep Q-Network model
    self.model = self._build_model()
    self.target_model = self._build_model()
    self.update_target_model()

    # Training parameters
    self.epsilon = EPSILON
    self.epsilon_decay = EPSILON_DECAY
    self.epsilon_min = MIN_EPSILON

  def _build_model(self):
      model = keras.Sequential([
          layers.Dense(128, input_shape=(self.state_size,) , activation='relu'),
          layers.Dense(128, activation='relu'),
          layers.Dense(self.action_size, activation='linear')  # Output Q-values for each action
      ])
      model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                    loss='mse')  # Mean Squared Error for Q-value approximation
      return model

  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    
    q_values = self.model.predict(state[np.newaxis, :], verbose=0)
    return np.argmax(q_values[0])
  
  def remember(self, state, action, reward, next_state, done):
    self.memory.add((state, action, reward, next_state, done))

  def replay_short_memory(self, state, action, reward, next_state, done):
      q_values = self.model.predict(state[np.newaxis, :], verbose=0)
      q_next_values = self.target_model.predict(next_state[np.newaxis, :], verbose=0)

      target = reward
      if not done:
          target += self.gamma * np.max(q_next_values[0])
      
      q_values[0][action] = target
      self.model.fit(state[np.newaxis, :], q_values, verbose=0)

  def replay_long_memory(self):
      if self.memory.size() < self.batch_size:
          return
      experiences = self.memory.sample(self.batch_size)
      states, actions, rewards, next_states, dones = zip(*experiences)

      states = np.array(states)
      next_states = np.array(next_states)
      rewards = np.array(rewards)
      dones = np.array(dones)

      # Current Q-values
      q_values = self.model.predict(states, verbose=0)
      q_next_values = self.target_model.predict(next_states, verbose=0)

      for i in range(self.batch_size):
          target = rewards[i]
          if not dones[i]:
              target += self.gamma * np.max(q_next_values[i])
          q_values[i][actions[i]] = target

      self.model.fit(states, q_values, verbose=0)

  def decay_epsilon(self):
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

  def save_model(self, filename):
    self.model.save(filename)