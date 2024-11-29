import random

import numpy as np
import pygame
from gymnasium import spaces, Env


def turn_left(direction):
  turn_left_map = {'UP': 'LEFT', 'LEFT': 'DOWN', 'DOWN': 'RIGHT', 'RIGHT': 'UP'}
  return turn_left_map[direction]


def turn_right(direction):
  turn_right_map = {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP'}
  return turn_right_map[direction]


class SnakeEnv(Env):
  metadata = {"render_modes": ["human"], "render_fps": 30}

  def __init__(self, cell_size=20, max_steps=200, render_mode=None):
    super(SnakeEnv, self).__init__()

    self.grid_size = 0
    self.cell_size = cell_size
    self.max_steps = max_steps
    self.render_mode = render_mode

    self.action_space = spaces.Discrete(3)  # Action space: 0 = straight, 1 = left, 2 = right

    self.score = 0
    self.snake = []
    self.done = False
    self.apple = None
    self.steps_taken = 0
    self.direction = "UP"

    # Pygame initialization
    self.window = None
    self.clock = None

    self.window_size = 0

  def step(self, action):
    self.steps_taken += 1

    if action == 1:  # Turn left
      self.direction = turn_left(self.direction)
    elif action == 2:  # Turn right
      self.direction = turn_right(self.direction)

    self._move()
    reward = -0.01

    if self._check_collision():
      self.done = True
      reward = -10
    elif self.snake[0] == self.apple:
      self.snake.append(self.snake[-1])
      self.apple = self._place_apple()
      reward = 10
      self.score += 1

    if self.steps_taken >= self.max_steps:
      self.done = True

    return self._get_observation(), reward, self.done, {"score": self.score}

  def render(self, mode='human'):
    if self.window is None:
      pygame.init()
      self.window = pygame.display.set_mode((self.window_size, self.window_size))
      pygame.display.set_caption("Snake")
      self.clock = pygame.time.Clock()

    self.window.fill((0, 0, 0))  # Black background

    self._custom_render()

    # Draw the snake
    for segment in self.snake:
      pygame.draw.rect(
        self.window,
        (0, 255, 0),  # Green color
        pygame.Rect(segment[1] * self.cell_size, segment[0] * self.cell_size, self.cell_size, self.cell_size)
      )

    # Draw the apple
    pygame.draw.rect(
      self.window,
      (255, 0, 0),  # Red color
      pygame.Rect(self.apple[1] * self.cell_size, self.apple[0] * self.cell_size, self.cell_size, self.cell_size)
    )

    pygame.display.flip()
    self.clock.tick(self.metadata["render_fps"])  # Limit the frame rate

  def close(self):
    pygame.quit()

  def _custom_render(self):
    pass

  def _get_observation(self) -> np.ndarray:
    pass

  def _is_outside_map(self, point):
    return point[0] < 0 or point[1] < 0 or point[0] >= self.grid_size or point[1] >= self.grid_size

  def _move(self):
    head = self.snake[0].copy()

    if self.direction == 'UP':
      head[1] -= 1
    elif self.direction == 'DOWN':
      head[1] += 1
    elif self.direction == 'LEFT':
      head[0] -= 1
    elif self.direction == 'RIGHT':
      head[0] += 1

    self.snake.insert(0, head)
    self.snake.pop()

  def _place_apple(self):
    while True:
      apple = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
      if apple not in self.snake:
        return apple

  def _check_collision(self):
    head = self.snake[0]
    return (head in self.snake[1:] or head[0] < 0 or head[1] < 0 or
            head[0] >= self.grid_size or head[1] >= self.grid_size)

  def _check_danger(self, point):
    return point in self.snake or self._is_outside_map(point)

  def _check_future_collision(self, direction):
    head = self.snake[0].copy()

    if direction == 'UP':
      head[1] -= 1
    elif direction == 'DOWN':
      head[1] += 1
    elif direction == 'LEFT':
      head[0] -= 1
    elif direction == 'RIGHT':
      head[0] += 1

    return self._check_danger(head)
