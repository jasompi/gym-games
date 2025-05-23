import random
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from gymnasium.spaces import MultiBinary, Discrete, Box


# Adapted from https://github.com/mbhenaff/PCPG/blob/main/deep_rl/component/envs.py
class DiabolicalCombLockEnv(gym.Env):
  ''' A diabolical combination lock environment: two locks
  In this task, an agent starts at an initial state s_0 (left most state), and based on its first action, transitions to one of two combination locks of length H. Each combination lock consists of a chain of length H, at the end of which are two states with high reward. At each level in the chain, 9 out of 10 actions lead the agent to a dead state (black) from which it cannot recover and lead to zero reward.
  Please check [PC-PG: Policy Cover Directed Exploration for Provable Policy Gradient Learning](http://arxiv.org/abs/2007.08459) for more details.
  '''
  def __init__(self, horizon=10, swap=0.5):
    super().__init__() # Initialize gym.Env
    self.horizon = horizon
    self.n_states = 3
    self.num_actions = 10
    self.n_locks = 2
    self.optimal_reward = 5.0
    self.suboptimal_reward = 2.0
    # Pass a unique seed or None to each sub-environment for independent initialization
    self.locks = [OneDiabolicalCombinationLock(horizon-1, seed_val=None), OneDiabolicalCombinationLock(horizon-1, seed_val=None)]
    self.action_space = Discrete(self.num_actions)
    self.n_features = self.locks[0].observation_space.shape[0] + 1
    self.observation_space = Box(low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32)
    # self.np_random for DiabolicalCombLockEnv itself, if needed for its own stochastic decisions
    # For now, its main stochasticity comes from sub-locks and action choices.

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed) # Important for wrappers and built-in seeding if any

    if seed is not None:
      # Determine reward assignment based on the main seed
      if seed % 2 == 0:
        self.locks[0].optimal_reward = self.optimal_reward
        self.locks[1].optimal_reward = self.suboptimal_reward
      else:
        self.locks[0].optimal_reward = self.suboptimal_reward
        self.locks[1].optimal_reward = self.optimal_reward
      
      self.locks[0].reset(seed=seed)
      self.locks[1].reset(seed=seed + 1) 

    self.h = 0
    obs = np.zeros(self.observation_space.shape, dtype=np.float32) # Ensure dtype
    return obs, {}

  def step(self, action):
    assert self.n_locks == 2
    terminated = False
    truncated = False # Assuming no explicit truncation condition here beyond horizon managed by sub-locks

    if self.h == 0:
      # In initial state, the action chooses lock
      self.lock_id = 0 if action < 5 else 1
      # Sub-lock reset should use its internally managed RNG state,
      # which was initialized/seeded when the main env was reset.
      # Or, if sub-lock needs fresh seeding at this point based on 'action', a derived seed could be passed.
      # For now, assume sub-lock's reset uses its existing RNG state.
      obs_lock, _ = self.locks[self.lock_id].reset() # sub-lock reset now returns (obs, info)
      reward = 0.0
      info = {'state': (0, self.h, self.lock_id)}
      obs = obs_lock # Use the observation from the sub-lock
    else:
      obs_lock, reward, term_lock, trunc_lock, info_lock = self.locks[self.lock_id].step(action)
      terminated = term_lock or trunc_lock # Combine termination & truncation from sub-lock
      info = info_lock # Use info from sub-lock
      info['state'] = info['state'] + (self.lock_id,)
      obs = obs_lock # Use the observation from the sub-lock
      
    self.h += 1
    # Append lock_id to the observation from the sub-lock
    # Ensure obs is an ndarray and the final obs is float32
    current_obs_is_ndarray = isinstance(obs, np.ndarray)
    if not current_obs_is_ndarray and obs is not None : # If obs is scalar or list, make it numpy array
        obs = np.array(obs, dtype=np.float32)
    elif obs is None: # If sub-lock returned None (e.g. terminal state)
        # This case needs careful handling. For now, creating a placeholder.
        # The observation space might not allow for None.
        # Fallback: create an array of zeros of the sub-lock's obs space shape.
        # This assumes sub-locks always have a defined obs space.
        # A more robust solution might involve a "is_terminal_obs" flag or similar.
        obs = np.zeros(self.locks[0].observation_space.shape, dtype=np.float32)

    obs = np.append(obs, float(self.lock_id)).astype(np.float32)


    # Check if DiabolicalCombLockEnv itself has a horizon that could lead to truncation
    # The horizon seems to be managed by the sub-locks as they are initialized with horizon-1
    # and their step function sets done = self.h == self.horizon (their local horizon)
    # So, `terminated` from sub-lock should be sufficient.

    return obs, reward, terminated, truncated, info

  def render(self, mode='human'):
    return (self.locks[0].render(mode), self.locks[1].render(mode))

  def close(self):
    self.locks[0].close()
    self.locks[1].close()
    return None


class OneDiabolicalCombinationLock(gym.Env):
  """ One Diabolical Stochastic Combination Lock
  :param horizon: Horizon of the MDP
  :param swap: Probability for stochastic edges
  """
  def __init__(self, horizon=10, swap=0.5, seed_val=None): # seed_val for initial RNG setup
    super().__init__() # Initialize gym.Env
    self.horizon = horizon
    self.swap = swap
    self.tolerance = 0.5
    self.optimal_reward = 5.0 # This can be changed by parent env
    self.optimal_reward_prob = 1.0
    self.anti_shaping_reward = 0.0
    self.anti_shaping_reward2 = 1.0
    assert self.anti_shaping_reward < self.optimal_reward * self.optimal_reward_prob, \
      "Anti shaping reward shouldn't exceed optimal reward which is %r" % \
      (self.optimal_reward * self.optimal_reward_prob)
    self.num_actions = 10
    self.actions = list(range(self.num_actions))
    self.action_space = gym.spaces.Discrete(self.num_actions)
    self.obs_dim = 2 * horizon + 4
    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
    self.np_random, _ = seeding.np_random(seed_val) # Initialize own RNG

  def render(self, mode='human'):
    return self.make_obs(self.state)

  def make_obs(self, x):
    if x is None or self.obs_dim is None:
      return x
    else:
      v = np.zeros(self.obs_dim, dtype=np.float32)
      v[x[0]] = 1.0
      v[3 + x[1]] = 1.0
      return v

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed) # Important for wrappers

    if seed is not None:
      # Re-initialize the RNG with the new seed
      self.np_random, _ = seeding.np_random(seed) # Corrected: re-initialize generator
      # Also re-initialize opt_a and opt_b as they depend on the RNG
      self.opt_a = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon)
      self.opt_b = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon)
    
    # Initialize opt_a and opt_b if not already done (e.g., if seed was None on first call but RNG is now seeded)
    # This ensures they are set before the first step.
    if not hasattr(self, 'opt_a'):
        self.opt_a = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon)
        self.opt_b = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon)

    # Start stochastically in one of the two live states
    toss_value = self.np_random.binomial(1, 0.5)
    if toss_value == 0:
      self.state = [0, 0]
    elif toss_value == 1:
      self.state = [1, 0]
    else:
      raise AssertionError("Toss value can only be 1 or 0. Found %r" % toss_value)
    self.h = 0
    return self.make_obs(self.state), {}

  def transition(self, x, a):
    if x is None:
      raise Exception("Not in any state")
    b = self.np_random.binomial(1, self.swap)
    # Ensure opt_a and opt_b are initialized
    if not hasattr(self, 'opt_a') or not hasattr(self, 'opt_b'):
        # This might happen if reset(seed=None) was called first, then step.
        # It's better to ensure opt_a/opt_b are initialized in reset.
        # For safety, we can initialize them here if absolutely necessary,
        # but it indicates a potential logic flow issue if reset wasn't properly called/seeded.
        self.opt_a = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon)
        self.opt_b = self.np_random.integers(low=0, high=self.action_space.n, size=self.horizon)

    if x[0] == 0 and a == self.opt_a[x[1]]:
      if b == 0:
        return [0, x[1] + 1]
      else:
        return [1, x[1] + 1]
    if x[0] == 1 and a == self.opt_b[x[1]]: # Corrected from self.opt_a to self.opt_b
      if b == 0:
        return [1, x[1] + 1]
      else:
        return [0, x[1] + 1]
    else:
      return [2, x[1] + 1]

  def reward(self, x, a, next_x):
    # If the agent reaches the final live states then give it the optimal reward.
    if (x == [0, self.horizon-1] and a == self.opt_a[x[1]]) or (x == [1, self.horizon-1] and a == self.opt_b[x[1]]): # Corrected: self.opt_b
      return self.optimal_reward * self.np_random.binomial(1, self.optimal_reward_prob)
    # If reaching the dead state for the first time then give it a small anti-shaping reward.
    # This anti-shaping reward is anti-correlated with the optimal reward.
    if x is not None and next_x is not None:
      if x[0] != 2 and next_x[0] == 2:
        return self.anti_shaping_reward * self.np_random.binomial(1, 0.5)
      elif x[0] != 2 and next_x[0] != 2:
        return -self.anti_shaping_reward2 / (self.horizon-1) # Original logic, horizon-1 can be 0 if horizon=1
    return 0

  def step(self, action):
    if self.state is None:
      raise Exception("Episode is not started.")
    if self.h == self.horizon:
      new_state = None
    else:
      new_state = self.transition(self.state, action)
      self.h += 1
    reward_val = self.reward(self.state, action, new_state)
    self.state = new_state
    # Create a dictionary containing useful debugging information
    obs = self.make_obs(self.state)
    terminated = self.h == self.horizon # Episode ends if horizon is reached
    truncated = False # No other truncation condition specified
    info = {"state": None if self.state is None else tuple(self.state)}
    return obs, float(reward_val), terminated, truncated, info

  def close(self):
    return None


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)

if __name__ == '__main__':
  seed = 4
  set_random_seed(seed)
  env = DiabolicalCombLockEnv()
  env_cfg = {"horizon":5, "swap":0.5}
  # env.init(**env_cfg) # init is handled by __init__ now
  # env.seed(seed) # Seeding is handled by reset
  # For testing, explicitly reset with a seed
  obs, info = env.reset(seed=seed)
  
  # Seeding action space for deterministic action sampling in tests if needed
  # env.action_space.seed(seed) # This is how it would be done if action_space had a seed method
  # However, direct seeding of gym.spaces.prng (as in old code) is not good.
  # If env.action_space is a Discrete space, its sample() method might not be seedable post-initialization
  # without recreating the space or if it internally uses a seeded np_random instance.
  # For now, we rely on the default Discrete.sample() behavior.

  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  try:
    print('Obsevation space high:', env.observation_space.high)
    print('Obsevation space low:', env.observation_space.low)
  except:
    pass
  
  for i in range(1):
    # obs, info = env.reset(seed=seed+i) # Use different seeds for different episodes if desired for testing
    # The first reset is done above with the main seed.
    # If this loop is for multiple episodes, subsequent resets should also be handled correctly.
    # For this specific test structure, ob is already set from the first reset.
    print('Initial Observation:', obs) # obs from env.reset(seed=seed)
    while True:
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      print('Obser:', obs)
      print('action:', action)
      print('Reward:', reward)
      print('Terminated:', terminated)
      print('Truncated:', truncated)
      print('Done:', done)
      if done:
        break
  env.close()