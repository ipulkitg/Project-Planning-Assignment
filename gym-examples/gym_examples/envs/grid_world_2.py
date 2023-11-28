import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import time
import itertools
import random
from typing import Callable, NamedTuple, Optional, Union, List
from gymnasium.spaces.utils import flatten_space
from gymnasium.spaces import Dict,Discrete,Box

# Complications:
# issue 1 - How to contact between RL Agent and Bots.
# issue 2 - Propagate actions to all the Bots.
# issue 3 - 

# To do:
# 1. Setup a step function to take actions list as input and give actions list as output : Mostly done
# 2. Setup the fixed layout of our environment in our reset function : Mostly done
# 3. Setting up the action and observation spaces. : Mostly done
# 4. Setting up the reward function.


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
        self.task = -1

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
                self.currLoc = (self.currLoc[0], self.currLoc[1]-1)
            elif self.currLoc[1] < self.destLoc[1]:
                #Move Up
                self.currLoc = (self.currLoc[0], self.currLoc[1]+1)
            else:
                self.task = -1
                return True
                #We are done so remove first task from list  
        elif self.currLoc[0] > self.destLoc[0]: 
            #Move Left
            self.currLoc = (self.currLoc[0] - 1, self.currLoc[1])
        else:
            #Move Right
            self.currLoc = (self.currLoc[0] + 1, self.currLoc[1])

        self.charge -= step_charge_cost

        return False #Has not reached the destination yet


    def pickup_dropoff_mats(self,raw,rawBuf,procBuf):
        # Check and try to dropoff
        i = 0
        rem_storage = raw_limit - len(rawBuf)
        raw_l = []
        reward = 0

        for r in self.storage:
            if i == rem_storage:
                break
            if r == raw:
                raw_l.append(i)
                i += 1

        rawBuf += raw_l # +1
        reward = len(raw_l)

        #Check and try to pickup
        while procBuf and len(self.storage) != logistic_limit:
            mat = procBuf.pop()
            self.storage.append(mat)
            reward += 1

        return rawBuf, procBuf, reward

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
            if self.slots[charge][0] >= 100:          #If we are fully charged then free the slot and add the bot's id to id_list
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
        elif len(self.procBuf) == proc_limit: #if it is full return True
            #Negative Reward for not picking up materials
            return True
        elif self.proc_check:
            self.processing -= 1
        return False # Return False when proc_Buff is not full

        #Convert raw to proc

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        # self.bots = bots  #[(object bot,id)]
        # self.factories = factories
        # self.stations = stations

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        agentDict = {}
        agentNum = 'agent1'
        for i in range(3):
            agentDict[agentNum] = spaces.Box(0, size - 1, shape=(2,), dtype=int)
            agentNum = agentNum.replace(str(i+1),str(i+2)) 
            
        #print(agentDict)
        
        # Task Statuses: Indirectly observed through the task array
        # 1: Distances of the bot from various entities
        # 2: Bot Current charge
        # 3: The proc_Buff of the factories
        # 4: The type of raw materials required for each factory

        self.observation_space = Dict(
            {
                # "agent": spaces.Dict(agentDict),
                # "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # "factory": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # "recharge": spaces.Box(0, size - 1, shape=(2,), dtype=int)
                        "bot_1_factory_raw":  Discrete(9),
                        "bot_1_factory_inter":  Discrete(9),
                        "bot_1_factory_final":  Discrete(9),
                        "bot_1_factory_delivery":  Discrete(9),
                        "bot_1_recharge":  Discrete(9),
                        "bot_2_factory_raw":  Discrete(9),
                        "bot_2_factory_inter":  Discrete(9),
                        "bot_2_factory_final":  Discrete(9),
                        "bot_2_factory_delivery":  Discrete(9),
                        "bot_2_recharge":  Discrete(9),
                        "bot_3_factory_raw":  Discrete(9),
                        "bot_3_factory_inter":  Discrete(9),
                        "bot_3_factory_final":  Discrete(9),
                        "bot_3_factory_delivery":  Discrete(9),
                        "bot_3_recharge":  Discrete(9),
                        "bot_1_charge":  Discrete(101),
                        "bot_2_charge":  Discrete(101),
                        "bot_3_charge":  Discrete(101),
                        "factory_raw_input":  Discrete(5),
                        "factory_inter_input":  Discrete(5),
                        "factory_final_input":  Discrete(5),
                        "factory_delivery_input":  Discrete(5),
                        "factory_raw_procBuff":  Discrete(101),
                        "factory_inter_procBuff":  Discrete(101),
                        "factory_final_procBuff":  Discrete(101)
            }
        )

        #Observation Variables
        self.time = 0
        self.recharge_color = (111,111,222)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.MultiDiscrete(np.array([7,7,7]))

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # self._action_to_direction = {
        #     0: np.array([1, 0]), #RIGHT
        #     1: np.array([0, 1]), #UP
        #     2: np.array([-1, 0]), #LEFT
        #     3: np.array([0, -1]), #DOWN
        # }

        self.action_dict = {
            0: "go_to_raw",
            1: "go_to_inter",
            2: "go_to_final",
            3: "go_to_delivery",
            4: "go_to_recharge",
            5: "pickup_dropoff",
            6: "recharge"
        }

        self.location_dict = {
            0: (0,0), #Raw_factory
            1: (2,2), #inter_factory
            2: (0,4), #final_factory
            3: (4,4), #Delivery_factory
            4: (4,0) #Recharge_station
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
        obs = {}
        for i,bot in enumerate(self.bots_list):
            obs_keys = ""
            bot_loc = bot.currLoc
            factor_raw_loc = self.factory_raw.loc
            factor_inter_loc = self.factory_inter.loc
            factor_final_loc = self.factory_final.loc
            factor_delivery_loc = self.factory_delivery.loc
            recharge_loc = self.reacharge_station.loc
            obs_keys = "bot_" + str(i+1) + "_"
            bot_distances = {
                        "factory_raw": abs(bot_loc[0] - factor_raw_loc[0]) + abs(bot_loc[1] - factor_raw_loc[1]),
                        "factory_inter": abs(bot_loc[0] - factor_inter_loc[0]) + abs(bot_loc[1] - factor_inter_loc[1]),
                        "factory_final": abs(bot_loc[0] - factor_final_loc[0]) + abs(bot_loc[1] - factor_final_loc[1]),
                        "factory_delivery": abs(bot_loc[0] - factor_delivery_loc[0]) + abs(bot_loc[1] - factor_delivery_loc[1]),
                        "recharge": abs(bot_loc[0] - recharge_loc[0]) + abs(bot_loc[1] - recharge_loc[1])
                    }
            obs[obs_keys + "factory_raw"] = bot_distances["factory_raw"]
            obs[obs_keys + "factory_inter"] = bot_distances["factory_inter"]
            obs[obs_keys + "factory_final"] = bot_distances["factory_final"]
            obs[obs_keys + "factory_delivery"] = bot_distances["factory_delivery"]
            obs[obs_keys + "recharge"] = bot_distances["recharge"]
        

        obs["bot_1_charge"] = self.bots_list[0].charge
        obs["bot_2_charge"] = self.bots_list[1].charge
        obs["bot_3_charge"] = self.bots_list[2].charge

        factory_raw = {
                "factory_raw": self.factory_raw.raw,
                "factory_inter": self.factory_inter.raw,
                "factory_final": self.factory_final.raw,
                "factory_delivery": self.factory_delivery.raw
        }
        obs["factory_raw_input"] = factory_raw["factory_raw"]
        obs["factory_inter_input"] = factory_raw["factory_inter"]
        obs["factory_final_input"] = factory_raw["factory_final"]
        obs["factory_delivery_input"] = factory_raw["factory_delivery"]

        factory_proc_buff = {
                "factory_raw": int(len(self.factory_raw.procBuf)/self.factory_raw.procBuf_len*100),
                "factory_inter": int(len(self.factory_inter.procBuf)/self.factory_inter.procBuf_len*100),
                "factory_final": int(len(self.factory_final.procBuf)/self.factory_final.procBuf_len*100)
        }
        obs["factory_raw_procBuff"] = factory_proc_buff["factory_raw"]
        obs["factory_inter_procBuff"] = factory_proc_buff["factory_inter"]
        obs["factory_final_procBuff"] = factory_proc_buff["factory_final"]

        # print(obs)

        return obs

    def _get_info(self):
        # return {
        #     "distance": np.linalg.norm(
        #         self._agent_location - self._target_location, ord=1
        #     )
        # }
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # (self,raw,processed,loc,procTime,rawBuf_len,procBuf_len)

        self.factory_raw = Factory(0,1,(0,0),0,1,4) # 1 is our raw material, 2 is intermediate, 3 is final

        self.factory_inter = Factory(1,2,(2,2),4,4,4)

        self.factory_final = Factory(2,3,(0,4),2,4,8)

        self.factory_delivery = Factory(3,4,(4,4),0,8,1)

        self.reacharge_station = RechargeStation((4,0))

        #self.bots_list = [LogisticBot((2,0),1),LogisticBot((2,0),2),LogisticBot((2,0),3)] #Randomize bots start location

        self.bots_list = []

        for i in range(3):
            loc = (random.randint(0,4),random.randint(0,4))
            self.bots_list.append(LogisticBot(loc,i+1))

        self.time = 0

        observation = self._get_obs()
        info = {0 : -1,
                1 : -1,
                2 : -1}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, bots_task_list):
        terminated = False
        reward = 0

        for i,task in enumerate(bots_task_list):
            if self.action_dict[task] == "go_to_raw":
                if self.bots_list[i].currLoc == self.location_dict[task]:
                    bots_task_list[i] = -1
                    reward -= 1
                    break
                self.bots_list[i].destLoc = self.location_dict[task]
                self.bots_list[i].task = task
                
                #small neg reward if trying to go to same location
            elif self.action_dict[task] == "go_to_inter":
                if self.bots_list[i].currLoc == self.location_dict[task]:
                    bots_task_list[i] = -1
                    reward -= 1
                    break
                self.bots_list[i].destLoc = self.location_dict[task]
                self.bots_list[i].task = task
                #small neg reward if trying to go to same location
            elif self.action_dict[task] == "go_to_final":
                if self.bots_list[i].currLoc == self.location_dict[task]:
                    bots_task_list[i] = -1
                    reward -= 1
                    break
                self.bots_list[i].destLoc = self.location_dict[task]
                self.bots_list[i].task = task
                #small neg reward if trying to go to same location
            elif self.action_dict[task] == "go_to_delivery":
                if self.bots_list[i].currLoc == self.location_dict[task]:
                    bots_task_list[i] = -1
                    reward -= 1
                    break
                self.bots_list[i].destLoc = self.location_dict[task]
                self.bots_list[i].task = task
                #small neg reward if trying to go to same location
            elif self.action_dict[task] == "go_to_recharge":
                if self.bots_list[i].currLoc == self.location_dict[task]:
                    bots_task_list[i] = -1
                    reward -= 1
                    break
                self.bots_list[i].destLoc = self.location_dict[task]
                self.bots_list[i].task = task
                #small neg reward if trying to go to same location
            elif self.action_dict[task] == "pickup_dropoff":
                pd_reward = 0
                if self.bots_list[i].currLoc == self.factory_raw.loc:
                    self.factory_raw.rawBuf,self.factory_raw.procBuf, pd_reward = self.bots_list[i].pickup_dropoff_mats(self.factory_raw.raw,self.factory_raw.rawBuf,self.factory_raw.procBuf)
                elif self.bots_list[i].currLoc == self.factory_inter.loc:
                    self.factory_inter.rawBuf,self.factory_inter.procBuf, pd_reward = self.bots_list[i].pickup_dropoff_mats(self.factory_inter.raw,self.factory_inter.rawBuf,self.factory_inter.procBuf)
                elif self.bots_list[i].currLoc == self.factory_final.loc:
                    self.factory_final.rawBuf,self.factory_final.procBuf, pd_reward = self.bots_list[i].pickup_dropoff_mats(self.factory_final.raw,self.factory_final.rawBuf,self.factory_final.procBuf)
                elif self.bots_list[i].currLoc == self.factory_delivery.loc:
                    self.factory_delivery.rawBuf,self.factory_delivery.procBuf, pd_reward = self.bots_list[i].pickup_dropoff_mats(self.factory_delivery.raw,self.factory_delivery.rawBuf,self.factory_delivery.procBuf)
                else:
                    self.bots_list[i].task = -1 #Do a small negative reward
                    bots_task_list[i] = -1
                    reward -= 8
                    break
                self.bots_list[i].task = -1 #Do a small positive reward
                bots_task_list[i] = -1
                reward += pd_reward
            elif self.action_dict[task] == "recharge":
                if self.bots_list[i].currLoc == self.reacharge_station.loc:
                    self.reacharge_station.put_to_charge(self.bots_list[i].charge,self.bots_list[i].id) #100-self.bots_list[i].charge will give the scale of how much reward to give
                    reward += (100-self.bots_list[i].charge)/20
                    self.bots_list[i].task = task #Do a scale positive reward
                else:
                    self.bots_list[i].task = -1 #Do a small negative reward
                    reward -= 1
            
        for i,bots in enumerate(self.bots_list): # Execute movement for each bot and then check if it is completed
            if bots.task < 5 and bots.task >= 0: # If its a movement task then do movement
                if bots.plan_movement():
                    bots_task_list[i] = -1
                    reward += 2
                    bots.task = -1
        
        bots_id_list = self.reacharge_station.charge()

        if bots_id_list:
            for bots in bots_id_list:
                self.bots_list[bots-1].task = -1
                bots_task_list[bots-1] = -1

        self.factory_raw.rawBuf = [0]
        if self.factory_raw.raw_to_proc():
            # Small neg reward
            reward -= 1
            pass
        if self.factory_inter.raw_to_proc():
            # Small neg reward
            reward -= 1
            pass
        if self.factory_final.raw_to_proc():
            # Small neg reward
            reward -= 1
            pass
        delivered_len = len(self.factory_delivery.rawBuf)
        self.factory_delivery.rawBuf = [] # Calculate reward based on delivered_len

        reward += (delivered_len*4)
        for bots in self.bots_list:
            if bots.charge <= 0:
                # Big neg reward
                reward -= 10
                terminated = True


        observation = self._get_obs()
        info = {0 : bots_task_list[0],
                1 : bots_task_list[1],
                2 : bots_task_list[2]}

        self.time += 1
        #Terminate when time has reached over episode time limit
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

        # Factory Raw
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * np.array([self.factory_raw.loc[0],self.factory_raw.loc[1]]), # Multiply with the location
                (pix_square_size, pix_square_size),
            ),
        )

        # Factory Inter
        pygame.draw.rect(
            canvas,
            (0, 255, 255),
            pygame.Rect(
                pix_square_size * np.array([self.factory_inter.loc[0],self.factory_inter.loc[1]]), # Multiply with the location
                (pix_square_size, pix_square_size),
            ),
        )

        # Factory Final
        pygame.draw.rect(
            canvas,
            (0, 255, 255),
            pygame.Rect(
                pix_square_size * np.array([self.factory_final.loc[0],self.factory_final.loc[1]]), # Multiply with the location
                (pix_square_size, pix_square_size),
            ),
        )

        # Factory Delivery
        pygame.draw.rect(
            canvas,
            (128, 0, 0),
            pygame.Rect(
                pix_square_size * np.array([self.factory_delivery.loc[0],self.factory_delivery.loc[1]]), # Multiply with the location
                (pix_square_size, pix_square_size),
            ),
        )

        #Recharge
        pygame.draw.rect(
            canvas,
            self.recharge_color,
            pygame.Rect(
                pix_square_size * np.array([self.reacharge_station.loc[0],self.reacharge_station.loc[1]]),
                (pix_square_size, pix_square_size),
            ),
        )
        if self.bots_list[0].charge <= 0:
            bot_colour = (10,10,10)
        else:
            bot_colour = (0,0,255)
        # Now we draw the agent 1
        pygame.draw.circle(
            canvas,
            bot_colour,
            (np.array([self.bots_list[0].currLoc[0],self.bots_list[0].currLoc[1]]) + 0.25) * pix_square_size,
            pix_square_size / 6,
        )
        if self.bots_list[1].charge <= 0:
            bot_colour = (10,10,10)
        else:
            bot_colour = (255,100,0)
        # Now we draw the agent 2
        pygame.draw.circle(
            canvas,
            bot_colour,
            (np.array([self.bots_list[1].currLoc[0],self.bots_list[1].currLoc[1]]) + 0.5) * pix_square_size,
            pix_square_size / 6,
        )
        if self.bots_list[2].charge <= 0:
            bot_colour = (10,10,10)
        else:
            bot_colour = (255,0,255)
        # Now we draw the agent 3
        pygame.draw.circle(
            canvas,
            bot_colour,
            (np.array([self.bots_list[2].currLoc[0],self.bots_list[2].currLoc[1]]) + 0.75) * pix_square_size,
            pix_square_size / 6,
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
    #print(observation)

    env.render_mode == "human"

    duaration = 5 # time - seconds
    framerate = 20 # FPS
    reward_array = []
    reward_sum = 0
    frames = [] # Empty list to append frames to
    bots_task_list = [-1,-1,-1]
    task_list = []
    for _ in range(500):
        for task in range(len(bots_task_list)):
            if info[task] == -1:
                bots_task_list[task] = np.random.randint(0,6)
        task_list.append(bots_task_list.copy()) 
        #print(bots_task_list) 
        #bots_task_list = [np.random.randint(0,4),np.random.randint(0,4),np.random.randint(0,4)]  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(bots_task_list) #2d array []
        px = env._render_frame()
        frames.append(px)
        reward_array.append(reward)
        reward_sum += reward
    #  media.show_image(env.render())
        if terminated or truncated:
            observation, info = env.reset()
            #print("Terminated")

    env.close()
    #print(frames)
    #print(np.random.randint(0,4))
    # print(task_list)
    # print(reward_array)
    # print(reward_sum)
    media.write_video('temp/videotrial.mp4', frames, fps=1)
