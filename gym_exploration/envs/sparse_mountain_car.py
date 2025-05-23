import math
import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv


class SparseMountainCarEnv(MountainCarEnv):
  ''' Modified based on Mountain Car.
    The only difference is the reward function: 
    the agent gets 0 reward every step until it reaches the goal with 1 reward.
  '''
  def __init__(self, goal_velocity=0):
    super().__init__(goal_velocity=goal_velocity)
  
  def step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    position, velocity = self.state
    velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
    velocity = np.clip(velocity, -self.max_speed, self.max_speed)
    position += velocity
    position = np.clip(position, self.min_position, self.max_position)
    if (position == self.min_position and velocity < 0):
        velocity = 0

    done = bool(
        position >= self.goal_position and velocity >= self.goal_velocity
    )
    # Set a sparse reward signal
    if done:
      reward = 1.0
    else:
      reward = 0.0

    self.state = (position, velocity)
    terminated = done # In MountainCar, 'done' usually means termination (goal reached)
    truncated = False # MountainCar is often wrapped with TimeLimit for truncation
    return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}



if __name__ == '__main__':
  env = SparseMountainCarEnv()
  # env.seed(0) # Old API; seeding is now done via reset
  obs, info = env.reset(seed=0) # Initial reset with seed

  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    # obs, info = env.reset(seed=i) # Reset for each episode if desired, obs is already from initial reset
    print('Initial Observation:', obs)
    for _ in range(10): # MountainCar can take many steps, 10 is just for a quick test
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