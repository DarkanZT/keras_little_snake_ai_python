import numpy as np
from keras import Model, optimizers
from numpy import ndarray, intp

from replay_buffer import ReplayBuffer

GAMMA = 0.99  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
BATCH_SIZE = 1000
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Frequency of updating the target network
LR = 0.001


class Agent:
  def __init__(self, action_size, gamma=GAMMA, lr=LR, batch_size=BATCH_SIZE, memory_size=BUFFER_SIZE,
               target_update_freq=TARGET_UPDATE_FREQ):
    self.action_size = action_size
    self.gamma = gamma  # Discount factor
    self.lr = lr  # Learning rate
    self.batch_size = batch_size
    self.memory_size = memory_size
    self.target_update_freq = target_update_freq

    self.n_games = 0

    self.memory = ReplayBuffer(self.memory_size)

    self.model = self._build_model()
    self.target_model = self._build_model()
    self.update_target_model()

    # Training parameters
    self.epsilon = EPSILON
    self.epsilon_decay = EPSILON_DECAY
    self.epsilon_min = MIN_EPSILON

  def _build_model(self) -> Model:
    pass

  def _choose_action(self, state: ndarray) -> intp:
    pass

  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  def act(self, state: ndarray) -> (int | intp):
    pass

  def remember(self, state: ndarray, action: (int | intp), reward: float, next_state: ndarray, done: bool):
    self.memory.add((state, action, reward, next_state, done))

  def replay_short_memory(self, state: ndarray, action: (int | intp), reward: float, next_state: ndarray, done: bool):
    pass

  def replay_long_memory(self):
    pass

  def decay_epsilon(self):
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

  def save_model(self, filename: str):
    self.model.save(filename)


def compile_model(model: Model, lr: float, loss: str = 'mse'):
  model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss)


def process_learning(agent: Agent, states, actions, rewards, next_states, dones):
  rewards = np.array(rewards)
  dones = np.array(dones)

  q_values = agent.model.predict(states, verbose=0)
  q_next_values = agent.model.predict(next_states, verbose=0)

  for i in range(agent.batch_size):
    target = rewards[i]
    if not dones[i]:
      target += agent.gamma * np.max(q_next_values[i])

    q_values[i][actions[i]] = target

  agent.model.fit(states, q_values, verbose=0)
