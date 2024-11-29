import random

import keras
import numpy as np
from keras import layers

import agent
from agent import Agent, compile_model


def choose_action(model: keras.Model, state: np.ndarray):
  q_values = model.predict(state[np.newaxis, :], verbose=0)
  return np.argmax(q_values[0])


class SimpleDQNAgent(Agent):
  def __init__(self, state_size, action_size, gamma=agent.GAMMA, lr=agent.LR, batch_size=agent.BATCH_SIZE,
               memory_size=agent.BUFFER_SIZE, target_update_freq=agent.TARGET_UPDATE_FREQ):
    self.state_size = state_size
    super(SimpleDQNAgent, self).__init__(action_size, gamma, lr, batch_size, memory_size, target_update_freq)

  def _build_model(self):
    model = keras.Sequential([
      layers.Dense(256, input_shape=(self.state_size,), activation='relu'),
      layers.Dense(256, activation='relu'),
      layers.Dense(self.action_size, activation='linear')  # Output Q-values for each action
    ])

    compile_model(model, self.lr)

    return model

  def _choose_action(self, state):
    return choose_action(self.model, state)

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)

    return self._choose_action(state)

  def replay_short_memory(self, state, action, reward, next_state, done):
    # Learn from a single state

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

    # Sample a batch of experiences from the memory and learn from them
    experiences = self.memory.sample(self.batch_size)
    states, actions, rewards, next_states, dones = zip(*experiences)

    states = np.array(states)
    next_states = np.array(next_states)

    agent.process_learning(self, states, actions, rewards, next_states, dones)
