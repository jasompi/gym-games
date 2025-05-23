import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from gymnasium.spaces import MultiBinary, Discrete, Box


# Adapted from https://raw.githubusercontent.com/microsoft/StateDecoding/master/LockBernoulli.py
class LockBernoulliEnv(gym.Env):
  ''' A (stochastic) combination lock environment
  You may configure the length, dimension, and switching probability.
  Check [Provably efficient RL with Rich Observations via Latent State Decoding](https://arxiv.org/pdf/1901.09018.pdf) for a detailed description.
  '''
  def __init__(self, dimension=0, switch=0.0, horizon=2):
    super().__init__()
    self.dimension = dimension
    self.switch = switch
    self.horizon = horizon
    self.n = self.dimension+3
    self.observation_space = Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)
    self.action_space = Discrete(4)
    self._seed() # Initialize np_random and opt_a, opt_b

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:
      self._seed(seed)
    
    self.h = 0
    self.state = 0
    obs = self.make_obs(self.state)
    return obs, {}

  def make_obs(self, s):
    new_x = np.zeros((self.n,), dtype=np.float32) # Ensure float32
    new_x[s] = 1.0 # Ensure float value
    # np_random.binomial returns int, but it's fine as it's assigned to a float32 array slices.
    # For explicit safety, could cast: .astype(np.float32)
    new_x[3:] = self.np_random.binomial(1, 0.5, (self.dimension,)).astype(np.float32)
    return new_x

  def step(self,action):
    assert self.h < self.horizon, 'Exceeded horizon!'
    terminated = False
    truncated = False
    
    if self.h == self.horizon-1:
      terminated = True
      r = self.np_random.binomial(1, 0.5)
      if self.state == 0 and action == self.opt_a[self.h]:
        next_state = 0
      elif self.state == 0 and action == (self.opt_a[self.h]+1) % 4:
        next_state = 1
      elif self.state == 1 and action == self.opt_b[self.h]:
        next_state = 1
      elif self.state == 1 and action == (self.opt_b[self.h]+1) % 4:
        next_state = 0
      else:
        next_state, r = 2, 0
    else:
      r = 0
      # terminated remains False
      ber = self.np_random.binomial(1, self.switch)
      if self.state == 0: # state A
        if action == self.opt_a[self.h]:
          next_state = ber
        elif action == (self.opt_a[self.h]+1) % 4:
          next_state = 1 - ber
        else:
          next_state = 2
      elif self.state == 1: # state B
        if action == self.opt_b[self.h]:
          next_state = 1 - ber
        elif action == (self.opt_b[self.h]+1) % 4:
          next_state = ber
        else:
          next_state = 2
      else: # state C
        next_state = 2

    self.h += 1
    self.state = next_state
    obs = self.make_obs(self.state)
    return obs, r, terminated, truncated, {}

  def render(self, mode='human'):
    print(f'{chr(self.state+65)}{self.h}')

  def _seed(self, seed=None): # Renamed from seed to _seed
    # Re-initialize the RNG with the new seed if provided, else it uses a random seed or existing state
    self.np_random, actual_seed = seeding.np_random(seed) 
    self.opt_a = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon) # Use integers for newer numpy
    self.opt_b = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon) # Use integers for newer numpy
    # Removed: if hasattr(gym.spaces, 'prng'): gym.spaces.prng.seed(seed)
    return seed # Though Gymnasium's reset doesn't expect seed to be returned from here

  def close(self):
    return 0


if __name__ == '__main__':
  env = LockBernoulliEnv()
  env.seed(0)
  env_cfg = {"horizon":10, "dimension":10, "switch":0.1}
  # env.init(**env_cfg) # __init__ now handles initialization logic
  # env.seed(0) is now env.reset(seed=0)
  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  try:
    print('Obsevation space high:', env.observation_space.high)
    print('Obsevation space low:', env.observation_space.low)
  except:
    pass

  for i in range(1):
    obs, info = env.reset(seed=0) # Updated reset call
    print('Observation:', obs)
    while True:
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action) # Updated step call
      done = terminated or truncated
      print('Observation:', obs)
      print('Reward:', reward)
      print('Terminated:', terminated)
      print('Truncated:', truncated)
      print('Done:', done)
      if done:
        break
  env.close()