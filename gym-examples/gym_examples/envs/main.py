import grid_world as gw
import numpy as np
import mediapy as media
import random

if __name__ == '__main__':
    
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    env = gw.GridWorldEnv()

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
            if bots_task_list[task] == -1:
                bots_task_list[task] = np.random.randint(0,6)
        task_list.append(bots_task_list.copy()) 
        print(bots_task_list) 
        #bots_task_list = [np.random.randint(0,4),np.random.randint(0,4),np.random.randint(0,4)]  # this is where you would insert your policy
        observation, reward, terminated, truncated, info, bots_task_list = env.step(bots_task_list) #2d array []
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
    print(task_list)
    print(reward_array)
    print(reward_sum)
    media.write_video('temp/videotrial.mp4', frames, fps=1)