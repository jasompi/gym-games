import os
import importlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ple import PLE


class BaseEnv(gym.Env):
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30} # Corrected metadata key

  def __init__(self, game_name='DefaultGame', normalize=False, display=False, **kwargs): # game_name must be passed by subclasses
    super().__init__()
    self.game_name = game_name # This should be set by the subclass before calling super().__init__ or this method
    
    game_module_name = f'ple.games.{self.game_name.lower()}'
    game_module = importlib.import_module(game_module_name)
    # PLE game kwargs are passed here.
    self.game = getattr(game_module, self.game_name)(**kwargs.get('ple_game_kwargs', {})) 

    if display == False:
      # Do not open a PyGame window
      os.putenv('SDL_VIDEODRIVER', 'fbcon')
      os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    self.normalize = normalize
    if self.normalize:
        self.state_processor = self.get_ob_normalize
    else:
        self.state_processor = self.get_ob
        
    # PLE handles its own display_screen logic based on SDL_VIDEODRIVER
    # Set state_preprocessor=None so gameOb.getGameState() returns the raw dict
    self.gameOb = PLE(self.game, fps=30, state_preprocessor=None, display_screen=display)
    
    self.viewer = None
    self.action_set = self.gameOb.getActionSet()
    self.action_space = spaces.Discrete(len(self.action_set))
    
    # Determine observation space shape from a sample processed observation
    # Get raw state from gameOb (which should be same as self.game.getGameState() now)
    _sample_raw_state = self.gameOb.getGameState() 
    _sample_processed_state = self.state_processor(_sample_raw_state)
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=_sample_processed_state.shape, dtype=np.float32)
    
    self.gameOb.init() # Initialize PLE system

  def get_ob(self, state_dict): # state_dict from game.getGameState()
    return np.array(list(state_dict.values()), dtype=np.float32)

  def get_ob_normalize(self, state_dict):
    # This method should be overridden by subclasses if normalize=True
    # For now, returning non-normalized as a fallback if not implemented by subclass.
    # Or raise NotImplementedError as before if strictness is required.
    # raise NotImplementedError('Get observation normalize function is not implemented by subclass!')
    return self.get_ob(state_dict)


  def step(self, action):
    reward = self.gameOb.act(self.action_set[action])
    terminated = self.gameOb.game_over()
    truncated = False # PLE environments don't typically have a separate truncation
    
    raw_state = self.gameOb.getGameState()
    observation = self.state_processor(raw_state)
    
    return observation, reward, terminated, truncated, {}
    
  def reset(self, *, seed: int | None = None, options: dict | None = None):
    super().reset(seed=seed)
    if seed is not None:
      self.gameOb.rng.seed(seed) # Seed PLE's internal RNG
      # PLE.init() re-initializes the game and its RNG.
      # We need to ensure the provided seed is used for the game logic.
      # The original PLE.init() calls self.game.init() which should use the seeded RNG.
      # If the game itself needs direct re-seeding, that would be more complex.
      # For now, relying on gameOb.rng.seed() and subsequent gameOb.reset_game().
      
    self.gameOb.reset_game()
    raw_state = self.gameOb.getGameState()
    observation = self.state_processor(raw_state)
    return observation, {}
  
  # Old seed method removed

  def render(self, mode='human'):
    # img = self.gameOb.getScreenRGB() 
    # img = self.gameOb.getScreenGrayscale()
    img = np.fliplr(np.rot90(self.gameOb.getScreenRGB(),3))
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      from gymnasium.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)

  def close(self):
    if self.viewer != None:
      self.viewer.close()
      self.viewer = None
    return 0