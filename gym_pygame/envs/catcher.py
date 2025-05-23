import os
import importlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ple import PLE

from gym_pygame.envs.base import BaseEnv


class CatcherEnv(BaseEnv):
  def __init__(self, normalize=True, display=False, **kwargs):
    self.game_name = 'Catcher' # Must be set before calling super().__init__
    # Pass ple_game_kwargs if any specific to Catcher game itself,
    # otherwise BaseEnv will pass an empty dict.
    super().__init__(game_name=self.game_name, normalize=normalize, display=display, **kwargs)
    
  def get_ob_normalize(self, state_dict): # Parameter name changed for clarity
    state_normal = self.get_ob(state_dict) # Use self.get_ob from BaseEnv
    state_normal = self.get_ob(state)
    state_normal[0] = (state_normal[0] - 26) / 26
    state_normal[1] = (state_normal[1]) / 8
    state_normal[2] = (state_normal[2] - 26) / 26
    state_normal[3] = (state_normal[3] - 20) / 45
    return state_normal

if __name__ == '__main__':
  env = CatcherEnv(normalize=True)
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
      # env.render('rgb_array')
      env.render('human')
      print('Observation:', obs)
      print('Reward:', reward)
      print('Terminated:', terminated)
      print('Truncated:', truncated)
      print('Done:', done)
      if done:
        break
  env.close()