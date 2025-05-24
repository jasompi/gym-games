import gymnasium as gym
import gym_pygame
import imageio
import numpy as np
import sys

env_id = sys.argv[1]
filename = sys.argv[2]

frame_buffer: list[np.ndarray] = []

env = gym.make(env_id, render_mode='rgb_array')
state, _ = env.reset()
frame_buffer.append(env.render())

for i in range(1000):
  action = env.action_space.sample()
  state, reward, done, truncated, _ = env.step(action)
  frame_buffer.append(env.render())
  if done or truncated:
    break

env.close()
imageio.mimwrite(filename, [np.array(img_frame) for _, img_frame in enumerate(frame_buffer)], fps=env.metadata['render_fps'])
