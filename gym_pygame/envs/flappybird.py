import os
import importlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ple import PLE

from gym_pygame.envs.base import BaseEnv


class FlappyBirdEnv(BaseEnv):
  def __init__(self, normalize=False, display=False, **kwargs):
    self.game_name = 'FlappyBird' # Must be set before calling super().__init__
    super().__init__(game_name=self.game_name, normalize=normalize, display=display, **kwargs)
    
  def get_ob_normalize(self, state_dict): # Changed from cls, state to self, state_dict
    state_normal = self.get_ob(state_dict) # Use self.get_ob from BaseEnv
    # TODO: Implement actual normalization for FlappyBird
    return state_normal

if __name__ == '__main__':
  env = FlappyBirdEnv(normalize=True) # normalize=True will try to use get_ob_normalize
  # env.seed(0) # Old API
  obs, info = env.reset(seed=0) # New API

  print('Action space:', env.action_space)
  # print('Action set:', env.action_set) # env.action_set is still available
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    # obs, info = env.reset(seed=i) # obs already from initial reset
    print('Initial Observation:', obs)
    while True:
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action) # Gymnasium API
      done = terminated or truncated
      env.render('human')
      #env.render('rgb_array')
      print('Observation:', obs)
      print('Reward:', reward)
      print('Terminated:', terminated)
      print('Truncated:', truncated)
      print('Done:', done)
      # break
      if done:
        break
  env.close()