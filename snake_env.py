import gymnasium as gym
import numpy as np
import pygame


class Env(gym.Env):
    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size=20):
        self.size = size
        self.window_size = 800

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Sequence(
                    gym.spaces.Box(0, size - 1, shape=(2,), dtype=int)
                ),
                "food": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        # return {"agent": self._snake_locations, "food": self._apple_location}
        world = np.zeros(shape=(self.size, self.size), dtype=np.float32)
        world[self._snake_locations[0][0], self._snake_locations[0][1]] = -2
        for square in self._snake_locations[1:]:
            world[square[0], square[1]] = -1
        world[self._apple_location[0], self._apple_location[1]] = 1

        # return tensor(world)
        return world

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._snake_locations = [np.array([self.size / 2, self.size / 2], dtype=int)]

        while True:
            self._apple_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

            for square in self._snake_locations:
                if np.array_equal(self._apple_location, square):
                    break
            else:
                break

        observation = self._get_obs()
        info = (self._snake_locations, self._apple_location)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        terminated = False

        direction = self._action_to_direction[action]
        self._snake_locations.insert(0, self._snake_locations[0] + direction)

        eat = np.array_equal(self._snake_locations[0], self._apple_location)
        if eat:
            if len(self._snake_locations) == self.size**2:
                terminated = True
            else:
                while True:
                    self._apple_location = self.np_random.integers(
                        0, self.size, size=2, dtype=int
                    )

                    for square in self._snake_locations:
                        if np.array_equal(self._apple_location, square):
                            break
                    else:
                        break
        else:
            self._snake_locations.pop()

        for square in self._snake_locations[1:]:
            if np.array_equal(self._snake_locations[0], square):
                collided = True
                break
        else:
            collided = False

        truncated = (
            not (
                0 <= self._snake_locations[0][0] < self.size
                and 0 <= self._snake_locations[0][1] < self.size
            )
            or collided
        )

        reward = 1 if eat else 0
        observation = None if truncated else self._get_obs()
        info = None if truncated else (self._snake_locations, self._apple_location)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        # if self.render_mode == "rgb_array":
        #     return self._render_frame()
        # elif self.render_mode == "human":
        #     return self._render_frame()
        return self._render_frame()

    def render_state(self, state):
        snake_locations, apple_location = state
        world = [(["- "] * self.size).copy() for _ in range(self.size)]
        world[snake_locations[0][0]][snake_locations[0][1]] = "H "
        for square in snake_locations[1:]:
            world[square[0]][square[1]] = "S "
        world[apple_location[0]][apple_location[1]] = "A "
        return (
            str(len(snake_locations)).center(self.size * 2)
            + "\n"
            + "\n".join(list(map("".join, world)))
        )

    def _render_frame(self):
        if self.render_mode == "ansi":
            world = [(["- "] * self.size).copy() for _ in range(self.size)]
            world[self._snake_locations[0][0]][self._snake_locations[0][1]] = "H "
            for square in self._snake_locations[1:]:
                world[square[0]][square[1]] = "S "
            world[self._apple_location[0]][self._apple_location[1]] = "A "
            return (
                str(len(self._snake_locations)).center(self.size * 2)
                + "\n"
                + "\n".join(list(map("".join, world)))
            )

        elif self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((0, 0, 0))
            # The size of a single grid square in pixels
            pix_square_size = self.window_size / self.size

            # First we draw the target
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (self._apple_location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
            # Now we draw the agent
            for square in self._snake_locations:
                pygame.draw.rect(
                    canvas,
                    (255, 255, 255),
                    pygame.Rect(
                        pix_square_size * square,
                        (pix_square_size, pix_square_size),
                    ),
                )

            if self.render_mode == "human":
                # The following line copies our drawings from `canvas` to the visible window
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()

                # We need to ensure that human-rendering occurs at the predefined framerate.
                # The following line will automatically add a delay to keep the framerate stable.
                self.clock.tick(self.metadata["render_fps"])
            else:  # rgb_array
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
