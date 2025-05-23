import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiBinary, Discrete, Box

from gym_exploration.envs.lockbernoulli import LockBernoulliEnv


class LockGaussianEnv(LockBernoulliEnv):
  ''' A (stochastic) combination lock environment
  The feature vector is hit with a random rotation and augmented with gaussian noise:
      x = s + eps where s is the one-hot encoding of the state.
  You may configure the length, dimension, and switching probability.
  Check [Provably efficient RL with Rich Observations via Latent State Decoding](https://arxiv.org/pdf/1901.09018.pdf) for a detailed description.
  '''
  def __init__(self, dimension=0, switch=0.0, noise=0.0, horizon=2):
    # Call the __init__ of the parent class (LockBernoulliEnv)
    super().__init__(dimension=dimension, switch=switch, horizon=horizon)
    self.noise = noise
  
  # The separate init method is no longer needed as its logic is merged into __init__
  # def init(self, dimension=0, switch=0.0, noise=0.0, horizon=2):
  #   super().init(horizon=horizon, dimension=dimension, switch=switch) # This was calling LockBernoulliEnv.init()
  #   self.noise = noise

  def make_obs(self, s):
    if self.noise > 0:
      new_x = np.random.normal(0, self.noise, [self.n]).astype(np.float32) # Ensure float32
    else:
      new_x = np.zeros((self.n,), dtype=np.float32) # Ensure float32
    new_x[s] += 1.0 # Ensure float value
    return new_x


if __name__ == '__main__':
  env = LockGaussianEnv()
  # env.seed(0) # Old API; seeding is now done via reset
  obs, info = env.reset(seed=0) # Initial reset with seed

  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    # obs, info = env.reset() # Reset for each episode if desired, obs is already from initial reset
    print('Initial Observation:', obs) 
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