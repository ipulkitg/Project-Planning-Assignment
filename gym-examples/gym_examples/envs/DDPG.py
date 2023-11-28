import gymnasium as gym
import numpy as np
import grid_world_3 as gw

from stable_baselines3 import DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gw.GridWorldEnv()

# The noise objects for DDPG
# n_actions = 7
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DQN("MultiInputPolicy", env, verbose=1)  # action_noise=action_noise, 
model.learn(total_timesteps=10000, log_interval=10)
model.save("ddpg_grid_world")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = DQN.load("ddpg_grid_world")

obs = vec_env.reset()
i = 0
while True or i < 1000:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    env.render("human")
    i += 1

# from stable_baselines3.common.env_checker import check_env
# import grid_world_2 as gw

# env = gw.GridWorldEnv()
# # It will check your custom environment and output additional warnings if needed
# check_env(env)