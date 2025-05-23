import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import seeding


# Adapted from https://github.com/facebookresearch/RandomizedValueFunctions/blob/master/qlearn/envs/nchain.py
class NChainEnv(gym.Env):
  ''' N-Chain environment
  The environment consists of a chain of N states and the agent always starts in state s2,
  from where it can either move left or right.
  In state s1, the agent receives a small reward of r = 0.001 and a larger reward r = 1 in state sN.
  Check [Deep Exploration via Bootstrapped DQN](https://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn.pdf) for a detailed description.
  '''
  def __init__(self, n=10):
    super().__init__()
    self.n = n
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)
    self.max_steps = n + 8
    self.state = 1  # Start at state s2
    self._seed() # Initialize np_random
    
  # Removed init method, merged into __init__
  
  def reward(self, s, a):
    if s == self.n-1 and a==1:
      return 1.0  
    elif s==0 and a==0:
      return 0.001
    else:
      return 0

  def step(self, action):
    assert self.action_space.contains(action)
    v = np.arange(self.n)
    
    reward_val = self.reward(self.state, action)
    if action == 1:
      if self.state != self.n - 1:
        self.state += 1
    else:
      if self.state != 0:
        self.state -= 1
    self.steps += 1
    
    terminated = False
    if self.state == 0 and action == 0: # Small reward state
        pass # Not necessarily terminal unless max_steps reached
    if self.state == self.n - 1 and action == 1: # Large reward state
        pass # Not necessarily terminal unless max_steps reached

    truncated = self.steps >= self.max_steps
    
    observation = (v <= self.state).astype('float32')
    return observation, reward_val, terminated, truncated, {}

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:
      self._seed(seed)
      
    v = np.arange(self.n)
    self.state = 1 # Start at state s2
    self.steps = 0
    return (v <= self.state).astype('float32'), {}
  
  def _seed(self, seed=None): # Renamed from seed
    # Re-initialize the RNG with the new seed if provided
    self.np_random, actual_seed = seeding.np_random(seed) 
    return actual_seed # Though Gymnasium's reset doesn't expect seed to be returned from here

  def render(self, mode='human'):
    pass

  def close(self):
    return 0
  

if __name__ == '__main__':
  env = NChainEnv()
  # env.seed(0) # Old API
  obs, info = env.reset(seed=0) # New API

  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  # Re-initialize with new config if needed for testing specific variant
  # This is more of a test setup concern than env API directly.
  # For Gymnasium, one might register different variants or pass args to make().
  # env_cfg = {'n':5}
  # env = NChainEnv(n=env_cfg['n']) # Re-initialize for new 'n'
  # obs, info = env.reset(seed=0) # Reset the new env

  print('Observation after potential re-init (if any):', obs)
  
  for i in range(1):
    # obs, info = env.reset(seed=i) # Reset for each episode if needed
    # obs is from the initial reset
    print('Initial Observation for episode:', obs)
    while True:
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action) # Gymnasium API
      done = terminated or truncated
      print('Observation:', obs)
      print('Reward:', reward)
      print('Terminated:', terminated)
      print('Truncated:', truncated)
      print('Done:', done)
      if done:
        break
  env.close()