import numpy as np
import pygame
import random
from gymnasium import spaces, Env

class SnakeEnv(Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10, cell_size=20, max_steps=200, render_mode=None):
        super(SnakeEnv, self).__init__()

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = self.grid_size * self.cell_size
        self.render_mode = render_mode

        self.max_steps = max_steps
        self.steps_taken = 0

        self.action_space = spaces.Discrete(3)  # Action space: 0 = straight, 1 = left, 2 = right
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        # Pygame initialization
        self.window = None
        self.clock = None

        self.reset()

    def reset(self):
        self.snake = [[self.grid_size // 2, self.grid_size // 2]]
        self.done = False
        self.direction = 'RIGHT'
        self.steps_taken = 0
        self.score = 0
        self.apple = self._place_apple()

        return self._get_observation()

    def step(self, action):
        self.steps_taken += 1

        if action == 1: # Turn left
            self.direction = self._turn_right(self.direction)
        elif action == 2: # Turn right
            self.direction = self._turn_right(self.direction)
        
        self._move()
        reward = 0

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
        self.clock.tick(self.metadata["render_fps"])  # Limit the frame rate to 10 FPS

    def close(self):
        pygame.quit()


    def _get_observation(self):
        head = self.snake[0]
        
        danger_straight = self._check_future_collision(self.direction)
        danger_left = self._check_future_collision(self._turn_left(self.direction))
        danger_right = self._check_future_collision(self._turn_right(self.direction))

        direction_flags = [
            self.direction == 'UP',
            self.direction == 'DOWN',
            self.direction == 'LEFT',
            self.direction == 'RIGHT',
        ]

        apple_direction = [
            self.apple[1] < head[1], # Apple is up
            self.apple[1] > head[1], # Apple is down
            self.apple[0] < head[0], # Apple is left
            self.apple[0] > head[0] # Apple is right
        ]

        return np.array([danger_straight, danger_left, danger_right] + direction_flags + apple_direction, dtype=np.float32)

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

        is_outside_map = head[0] < 0 or head[1] < 0 or head[0] >= self.grid_size or head[1] >= self.grid_size

        return head in self.snake or is_outside_map

    def _turn_left(self, direction):
        return {'UP': 'LEFT', 'LEFT': 'DOWN', 'DOWN': 'RIGHT', 'RIGHT': 'UP'}[direction]

    def _turn_right(self, direction):
        return {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP'}[direction]
