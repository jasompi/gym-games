import gymnasium as gym
import gym_minatar
import gym_pygame
import gym_exploration

class RandomAgent(object):
  def __init__(self, action_space):
    self.action_space = action_space

  def act(self, observation, reward, done):
    return self.action_space.sample()

if __name__ == '__main__':
  game = 'Catcher-PLE-v0'
  game = 'FlappyBird-PLE-v0'
  game = 'Pixelcopter-PLE-v0'
  game = 'PuckWorld-PLE-v0'
  game = 'Pong-PLE-v0'
  
  game = 'Asterix-MinAtar-v0'
  game = 'Breakout-MinAtar-v0'
  game = 'Freeway-MinAtar-v0'
  game = 'Seaquest-MinAtar-v0'
  game = 'SpaceInvaders-MinAtar-v0'

  game = 'Asterix-MinAtar-v1'
  game = 'Breakout-MinAtar-v1'
  game = 'Freeway-MinAtar-v1'
  game = 'Seaquest-MinAtar-v1'
  game = 'SpaceInvaders-MinAtar-v1'

  game = 'NChain-v1'
  game = 'LockBernoulli-v0'
  game = 'LockGaussian-v0'
  game = 'SparseMountainCar-v0'
  all_games = [
    'Catcher-PLE-v0',
    'FlappyBird-PLE-v0',
    'Pixelcopter-PLE-v0',
    'PuckWorld-PLE-v0',
    'Pong-PLE-v0',
    'Asterix-MinAtar-v0',
    'Breakout-MinAtar-v0',
    'Freeway-MinAtar-v0',
    'Seaquest-MinAtar-v0',
    'SpaceInvaders-MinAtar-v0',
    'Asterix-MinAtar-v1',
    'Breakout-MinAtar-v1',
    'Freeway-MinAtar-v1',
    'Seaquest-MinAtar-v1',
    'SpaceInvaders-MinAtar-v1',
    'NChain-v1',
    'LockBernoulli-v0',
    'LockGaussian-v0',
    'SparseMountainCar-v0',
    'DiabolicalCombLock-v0'
  ]

  for game_name in all_games:
    print(f"\nTesting game: {game_name}")
    try:
      env = gym.make(game_name)
    except Exception as e:
      print(f"ERROR creating environment {game_name}: {e}")
      continue # Skip to next game if creation fails

    # The env.init() pattern is not standard in Gymnasium.
  # Environment-specific configurations are typically passed via gym.make() or handled in the env's __init__.
  # For now, we'll remove these calls to allow the script to run.
  # If specific configurations from game_cfg are essential, the environment registration
  # or their __init__ methods would need to be adapted.
  # For example, for NChain-v1, if 'n' is a parameter to its __init__,
  # it might be possible to do gym.make('NChain-v1', n=5) if registered appropriately.
  # However, the current registrations do not show these args.
  # if game in ['NChain-v1', 'LockBernoulli-v0', 'LockGaussian-v0', 'DiabolicalCombLock-v0']:
  #   game_cfg = {
  #     'NChain-v1': {'n':5},
  #     'LockBernoulli-v0': {'horizon':10, 'dimension':10, 'switch':0.1},
  #     'LockGaussian-v0': {'horizon':9, 'dimension':9, 'switch':0.1, 'noise':0.1},
  #     'DiabolicalCombLock-v0': {"horizon":5, "swap":0.5}
  #   }
  #   # env.init(**game_cfg[game]) # This line is removed
  
  # Seeding in Gymnasium is done via env.reset(seed=...)
  # The action space can also be seeded if necessary, but it's less common for typical envs.
  # env.action_space.seed(0) # If action space seeding is critical.

    print('Game:', game_name) # Use game_name from loop
    print('Action space:', env.action_space)
    print('Obsevation space:', env.observation_space)
  try:
    print('Obsevation space high:', env.observation_space.high)
    print('Obsevation space low:', env.observation_space.low)
  except:
    pass

    for i in range(1): # Keep the single episode test for brevity
      # Initial reset with seed.
      try:
        ob, info = env.reset(seed=0) 
      except Exception as e:
        print(f"ERROR during env.reset() for {game_name}: {e}")
        break # Break from this episode's loop

      for _ in range(3): # Perform 3 steps
        try:
          action = env.action_space.sample()
          ob, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated
        except Exception as e:
          print(f"ERROR during env.step() for {game_name}: {e}")
          done = True # Assume error means episode is over for this test
        
        # env.render() # default render mode is 'human'
        # env.render('human') # Avoid rendering in automated tests to prevent display issues/hangs
        # img = env.render('rgb_array')
        if 'e' in locals() and e is not None: # if an error occurred in step
            break 
            
        print('Observation shape:', ob.shape if hasattr(ob, 'shape') else type(ob)) # Print shape or type
        print('Reward:', reward)
        print('Terminated:', terminated)
        print('Truncated:', truncated)
        print('Done:', done)
        if done:
          break
      if 'e' in locals() and e is not None: # if an error occurred in reset
          e = None # reset error for next game
          break

    try:
      env.close()
    except Exception as e:
      print(f"ERROR during env.close() for {game_name}: {e}")