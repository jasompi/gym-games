import gymnasium as gym
import gym_pygame # Ensure this is imported to register the envs
import numpy as np # Import numpy for a potential check, though not strictly in user script

print("Attempting to make Pixelcopter-PLE-v0...")
env = gym.make('Pixelcopter-PLE-v0', render_mode='human')
print("Environment made successfully.")

print("Resetting environment...")
state, info = env.reset(seed=42) # Use a seed for reproducibility
print(f"Reset successful. Initial state type: {type(state)}, shape: {state.shape if isinstance(state, np.ndarray) else 'N/A'}")

# Check if spec.max_episode_steps is available, otherwise use a default
max_steps = getattr(env.spec, 'max_episode_steps', 200) # Default to 200 if not found
if max_steps is None: # Handles if max_episode_steps is None
    max_steps = 200
print(f"Max episode steps: {max_steps}")

for i in range(max_steps):
  action = env.action_space.sample()
  # print(f"Step {i}, Action: {action}") # Optional: for verbose logging
  state, reward, terminated, truncated, info = env.step(action)
  # print(f"State type: {type(state)}, Shape: {state.shape if isinstance(state, np.ndarray) else 'N/A'}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}") # Optional
  if terminated or truncated:
    print(f"Episode finished after {i+1} steps. Terminated: {terminated}, Truncated: {truncated}")
    break
else: # Executed if loop finishes without break
    print(f"Episode reached max_steps ({max_steps}) without termination or truncation.")

env.close()
print("Test script completed.")
