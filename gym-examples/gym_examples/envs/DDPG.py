import gymnasium as gym
import numpy as np
import grid_world_3 as gw
import mediapy as media

from stable_baselines3 import DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gw.GridWorldEnv()

# The noise objects for DDPG
# n_actions = 7
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DQN("MultiInputPolicy", env, verbose=1)  # action_noise=action_noise, 
# model.learn(total_timesteps=100000, log_interval=10)
# model.save("ddpg_grid_world")
#vec_env = model.get_env()

# del model # remove to demonstrate saving and loading

model = DQN.load("ddpg_grid_world")

vec_env = model.get_env()
print(vec_env)
task_list = []
obs = vec_env.reset()
info = {0 : -1}
reward_array = []
frames = []
reward_sum = 0
for i in range(500):
    if info[0] == 8:
        obs, rewards, dones, info = vec_env.step(8)
    elif info[0] != -1:
        obs, rewards, dones, info = vec_env.step(action)
    else:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
    task_list.append(action)
    px = env._render_frame()
    frames.append(px)
    reward_array.append(rewards)
    reward_sum += rewards
    # env.render()
vec_env.close()

print(reward_array)
print("\n")
print("\n")
print("\n")
print("\n")
print(task_list)
print(reward_sum)

media.write_video('temp/videoDQN.mp4', frames, fps=10)



# from stable_baselines3.common.env_checker import check_env
# import grid_world_2 as gw

# env = gw.GridWorldEnv()
# # It will check your custom environment and output additional warnings if needed
# check_env(env)