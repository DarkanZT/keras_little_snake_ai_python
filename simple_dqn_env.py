import numpy as np
import pygame
from gymnasium import spaces

from snake_env import SnakeEnv

VISION_RANGE = 6


class SimpleDQNEnv(SnakeEnv):
  def __init__(self, grid_size=10, cell_size=20, max_steps=200, render_mode=None):
    super(SimpleDQNEnv, self).__init__(cell_size, max_steps, render_mode)

    self.grid_size = grid_size

    self.window_size = self.grid_size * self.cell_size

    state_length = VISION_RANGE * VISION_RANGE + 4 + 4  # 4 for apple_direction and for direction_flags

    self.observation_space = spaces.Box(low=0, high=1, shape=(state_length,), dtype=np.float32)

    self.reset()

  def reset(self):
    self.snake = [[self.grid_size // 2, self.grid_size // 2]]
    self.done = False
    self.direction = 'UP'
    self.steps_taken = 0
    self.score = 0
    self.apple = self._place_apple()

    return self._get_observation(), {}

  def _custom_render(self):
    self._render_vision(self.cell_size)

  def _render_vision(self, cell_size):
    head_x, head_y = self.snake[0]

    color = (50, 50, 50)  # Dark gray

    # Decode vision into coordinates
    for dx in range(-VISION_RANGE, VISION_RANGE + 1):
      for dy in range(-VISION_RANGE, VISION_RANGE + 1):
        nx, ny = head_x + dx, head_y + dy

        # Skip out-of-bounds cells
        if self._is_outside_map([nx, ny]):
          continue

        # Draw the vision cell
        rect = pygame.Rect(ny * cell_size, nx * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.window, color, rect)

  def _get_observation(self):
    head = self.snake[0]

    vision = self._get_vision()

    direction_flags = [
      self.direction == 'UP',
      self.direction == 'DOWN',
      self.direction == 'LEFT',
      self.direction == 'RIGHT',
    ]

    apple_direction = [
      self.apple[1] < head[1],  # Apple is up
      self.apple[1] > head[1],  # Apple is down
      self.apple[0] < head[0],  # Apple is left
      self.apple[0] > head[0]  # Apple is right
    ]

    return np.array(vision + direction_flags + apple_direction, dtype=np.float32)

  def _get_vision(self):
    """
    Get the vision of the snake.

    :return: Whether there is a danger in the vision range or not
    :rtype: list[bool]
    """

    head_x, head_y = self.snake[0]
    vision = []

    for dx in range(-VISION_RANGE // 2, VISION_RANGE // 2 + 1):
      for dy in range(-VISION_RANGE // 2, VISION_RANGE // 2 + 1):
        nx, ny = head_x + dx, head_y + dy

        vision.append(self._check_danger([nx, ny]))

    return vision
