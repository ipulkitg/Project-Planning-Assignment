import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import time
import itertools
from typing import Callable, NamedTuple, Optional, Union, List

# Complications:
# issue 1 - How to contact between RL Agent and Bots.
# issue 2 - Propagate actions to all the Bots.
# issue 3 - 


#Global Variables
step_charge_cost = 5 
step_recharge = 5
logistic_limit = 4
raw_limit = 4
proc_limit = 4
recharge_slots = 4
processing = 0

class LogisticBot:
    def __init__(self,currLoc,id) -> None:
        self.id = id
        self.currLoc = currLoc
        self.destLoc = currLoc
        self.tasks = [] 

        #List of possible tasks:
        # 1. Go to location
        #     a) Go to factory and carry out pickup/dropoff tasks  --- Possibly can split pickup and dropoff tasks
        #     b) Go to recharge station for charging
        # 2. Ask for more tasks
        #     a) If task list is empty
        #     b) If in the same location for a while (If waiting recharge station or if we implement stochastic factories for pickup and the bot is waiting in that location for a while)
        
        #task length
        self.poll = False #Implement as a function
        self.charge = 100
        self.storage = []
        self.storage_limit = logistic_limit

    def plan_movement(self):
        # Move Left or Right till you are on the same coloumn, then move Up or Down till you reach Destination
        if self.currLoc[0] == self.destLoc[0]: 
            if self.currLoc[1] > self.destLoc[1]:
                #Move Down
                self.currLoc[1] -= 1
            elif self.currLoc[1] < self.destLoc[1]:
                #Move Up
                self.currLoc[1] += 1
            else:
                return True
                #We are done so remove first task from list  
        elif self.currLoc[0] > self.destLoc[0]: 
            #Move Left
            self.currLoc[0] -= 1
        else:
            #Move Right
            self.currLoc[0] += 1

        self.charge -= step_charge_cost
        return False #Has not reached the destination yet

    def pickup_dropoff_mats(self,raw,rawBuf,procBuf):
        # Check and try to dropoff
        i = 0
        rem_storage = raw_limit - len(rawBuf)
        raw_l = []

        for r in self.storage:
            if i == rem_storage:
                break
            if r == raw:
                raw_l.append(i)
                i += 1

        rawBuf += raw_l # +1

        #Check and try to pickup
        while procBuf or len(self.storage) != logistic_limit:
            mat = procBuf.pop()
            self.storage.append(mat)

        #Check if you are on a factory and which factory
        #Check the amount of materials in the ProcBuff
    
class RechargeStation:
    def __init__(self,loc) -> None:
        self.slots = []
        self.loc = loc

    #When a bot reaches a station
    def put_to_charge(self,curr_charge,id):       
        if len(self.slots) != recharge_slots:
            self.slots.append([curr_charge,id]) #We are appending each id and their current charge

    #Every step 
    def charge(self):
        id_list = []                                  #List of IDs
        for charge in range(len(self.slots)):         
            self.slots[charge][0] += step_recharge
            if self.solts[charge][0] >= 100:          #If we are fully charged then free the slot and add the bot's id to id_list
                done = self.slots.pop(charge)
                id_list.append(done[1])
        return id_list                                #Return the list of IDs so that the environment can notify which bots are free to move (To be Done)

class Factory:
    def __init__(self,raw,processed,loc,procTime,rawBuf_len,procBuf_len) -> None:
        self.raw = raw 
        self.loc = loc
        self.processed = processed
        self.procTime = procTime
        self.rawBuf = []
        self.rawBuf_len = rawBuf_len
        self.procBuf = []
        self.procBuf_len = procBuf_len
        self.processing = procTime #Since each factory may process concurrently, a temp variable to count down to 0
        self.proc_check = False

    def raw_to_proc(self):
        if self.rawBuf and self.processing == self.procTime: #if there is raw materials in buffer and the prev raw material is not processing
            self.rawBuf.pop()
            self.proc_check = True
        elif len(self.procBuf) != proc_limit and self.processing <= 0: #if processing is over and proc buffer is not full
            self.procBuf.append(self.processed)
            self.processing == self.procTime
            self.proc_check = False
        elif len(self.procBuf) == proc_limit: #if it is full
            #Negative Reward for not picking up materials
            pass
        elif self.proc_check:
            self.processing -= 1

        #Convert raw to proc

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, bots, factories, stations, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.bots = bots
        self.factories = factories
        self.stations = stations

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        agentDict = {}
        agentNum = 'agent1'
        for i in range(3):
            agentDict[agentNum] = spaces.Box(0, size - 1, shape=(2,), dtype=int)
            agentNum = agentNum.replace(str(i+1),str(i+2)) 
            
        print(agentDict)
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Dict(agentDict),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "factory": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "recharge": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )

        #Observation Variables
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
        return ()

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

    def step(self):
        
        # Map the action (element of {0,1,2,3}) to the direction we walk in

        #Recharge work

        # # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )

        # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # reward = 1 if terminated else 0  # Binary sparse rewards

        reached = self.bots['id'].plan_movement()

        if reached:
            if self.bots['id'].currLoc == self.factories['id'].loc:
                self.factories['id'].pickup_dropoff_mats()
            elif self.bots['id'].currLoc == self.stations['id'].loc:
                self.stations['id'].put_to_charge(self.bots['id'].charge, self.bots['id'].id)

        #for all factories do raw_to_proc

        #for all stations do charge

        observation = self._get_obs()
        info = self._get_info()

        self.time += 1
        #Terminate when time has reached over episode time limit
        terminated = False
        if self.time >= 500:
            terminated = True
        
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

    bot = LogisticBot()

    for _ in range(1000):
        action = np.random.randint(0,4)  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(bot)
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
