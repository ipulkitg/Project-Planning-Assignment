import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import time
import itertools
from typing import Callable, NamedTuple, Optional, Union, List

class LogisticBot:
    def __init__(self,currLoc,destLoc) -> None:
        self.currLoc = currLoc
        self.destLoc = destLoc
        self.tasks = []
        self.poll = []
        self.charge = 100
    
class RechargeStation:
    def __init__(self) -> None:
        self.slots = [0,0,0,0]

class Factory:
    def __init__(self,raw,procTime) -> None:
        self.raw = raw
        self.procTime = procTime
        self.rawBuf = [0,0,0,0]
        self.procBuf = [0,0,0,0]

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "factory": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "recharge": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )

        #Observation Variables
        self.recharge_station = [0, 0, 0, 0]
        self.recharge_color = (255,255,0)
        self.recharge_time = 5
        self.time = 0

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]), #RIGHT
            1: np.array([0, 1]), #UP
            2: np.array([-1, 0]), #LEFT
            3: np.array([0, -1]), #DOWN
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        
        # Factory

        self._factory_location = self._agent_location
        while np.array_equal(self._factory_location, self._agent_location) or np.array_equal(self._factory_location, self._target_location):
            self._factory_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        #Recgarge

        self._recharge_location = self._agent_location
        while np.array_equal(self._recharge_location, self._agent_location) or np.array_equal(self._recharge_location, self._target_location) or np.array_equal(self._factory_location, self._recharge_location):
            self._recharge_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self.time = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = np.array([0,0])

        if self.recharge_station == [0,0,0,0]:
            direction = self._action_to_direction[action]
            self.recharge_color = (255,255,0)
            self.recharge_time = 5
        elif self.recharge_time == 0:
            self.recharge_station = [0,0,0,0]
            
        self.recharge_time -= 1

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        self.time += 1

        if np.array_equal(self._agent_location, self._recharge_location) and self.recharge_time > 0 :
            self.recharge_color = (0,255,255)
            self.recharge_station = [1,0,0,0]
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Factory
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._factory_location,
                (pix_square_size, pix_square_size),
            ),
        )
        #Recharge
        pygame.draw.rect(
            canvas,
            self.recharge_color,
            pygame.Rect(
                pix_square_size * self._recharge_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
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


if __name__ == '__main__':
    
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    env = GridWorldEnv()



    observation, info = env.reset()
    print(observation)

    env.render_mode == "human"

    duaration = 5 # time - seconds
    framerate = 20 # FPS

    frames = [] # Empty list to append frames to

    for _ in range(1000):
        action = np.random.randint(0,4)  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        px = env._render_frame()
        frames.append(px)

    #  media.show_image(env.render())
        if terminated or truncated:
            observation, info = env.reset()
            #print("Terminated")

    env.close()
    #print(frames)
    #print(np.random.randint(0,4))
    media.write_video('temp/videotrial.mp4', frames, fps=10)
