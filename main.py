from keras import models

from helper import plot, save_plot
from simple_dqn_agent import choose_action, SimpleDQNAgent
from simple_dqn_env import SimpleDQNEnv

NUM_EPISODES = 1000


def train_dqn(epochs=NUM_EPISODES, save_path="best_model.keras"):
  plot_scores = []
  plot_mean_scores = []
  best_score = 0
  total_score = 0

  env = SimpleDQNEnv(render_mode="human")

  # Environment and model setup
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n

  agent = SimpleDQNAgent(state_size, action_size)

  for epoch in range(epochs):
    state, _ = env.reset()
    done = False
    total_reward = 0
    info = {}

    while not done:
      action = agent.act(state)
      next_state, reward, done, info = env.step(action)
      total_reward += reward

      # Learn immediately from this single experience
      agent.replay_short_memory(state, action, reward, next_state, done)

      # Save experience for long-term replay
      agent.remember(state, action, reward, next_state, done)
      state = next_state

      # env.render()

      if done:
        agent.n_games += 1
        break

    # Learn from long memory
    agent.replay_long_memory()

    # Update target model
    if epoch % agent.target_update_freq == 0:
      agent.update_target_model()

    # Decay epsilon (exploration rate)
    agent.decay_epsilon()

    score = info["score"]

    if score >= best_score:
      best_score = score
      agent.save_model(save_path)  # Save the best performance

    plot_scores.append(score)
    total_score += score
    mean_score = total_score / agent.n_games
    plot_mean_scores.append(mean_score)
    plot(plot_scores, plot_mean_scores)

    print(f"Episode {epoch + 1}/{epochs}: Total Reward = {total_reward}: Score {score}: Epsilon = {agent.epsilon}")

  # env.close()
  # Save the plot only after all games are finished
  save_plot("final_plot.png")


if __name__ == "__main__":
  train_dqn()

  env = SimpleDQNEnv(grid_size=24, render_mode="human")

  trained_model = models.load_model("best_model.keras")

  state, _ = env.reset()
  done = False

  while not done:
    action = choose_action(trained_model, state)

    next_state, reward, done, info = env.step(action)

    state = next_state

    total_score = info["score"]
    env.render()  # Watch the snake in action

  print("Total Score in Evaluation:", total_score)
  env.close()
