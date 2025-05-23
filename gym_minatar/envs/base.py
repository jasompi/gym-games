import gymnasium as gym
from gymnasium import spaces

from minatar import Environment


class BaseEnv(gym.Env):
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30} # Corrected metadata key

  def __init__(self, game, display_time=50, use_minimal_action_set=False, **kwargs):
    super().__init__()
    self.game_name = game
    self.display_time = display_time
    self.game_kwargs = kwargs
    # MinAtar Environment does not take random_seed in __init__
    self.game = Environment(env_name=self.game_name, **self.game_kwargs)
    if use_minimal_action_set:
      self.action_set = self.game.minimal_action_set()
    else:
      self.action_set = list(range(self.game.num_actions()))
    self.action_space = spaces.Discrete(len(self.action_set))
    self.observation_space = spaces.Box(0.0, 1.0, shape=self.game.state_shape(), dtype=bool)

  def step(self, action):
    action = self.action_set[action]
    reward, terminated = self.game.act(action) # MinAtar's act returns reward, done
    truncated = False # MinAtar environments typically don't have a separate truncation condition
    return self.game.state(), reward, terminated, truncated, {}
    
  def reset(self, *, seed: int | None = None, options: dict | None = None):
    super().reset(seed=seed) # Call to gym.Env.reset()
    if seed is not None:
      # MinAtar's Environment is seeded using its own seed method.
      # No need to re-create the Environment object for seeding.
      self.game.seed(seed) 
      # Action set might change if game parameters change, though not typical with just seed.
      # Re-fetch action set to be safe, assuming minimal_action_set is stored/passed in game_kwargs or self
      if hasattr(self, 'use_minimal_action_set') and self.use_minimal_action_set: # Check if attr exists
          self.action_set = self.game.minimal_action_set()
      else:
          self.action_set = list(range(self.game.num_actions()))
      # self.action_space might need to be redefined if len(self.action_set) changes.
      # This is unlikely if only the seed changes.
      # For simplicity, assuming action space structure doesn't change with seed.
    
    self.game.reset() # Reset the game state
    return self.game.state(), {}
  
  # The old seed method is removed. Seeding is handled in reset().

  def render(self, mode='human'):
    if mode == 'rgb_array':
      return self.game.state()
    elif mode == 'human':
      self.game.display_state(self.display_time)

  def close(self):
    if self.game.visualized:
      self.game.close_display()
    return 0