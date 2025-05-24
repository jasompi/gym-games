import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import sys

# Import the custom envs so they are registered
import gym_pygame 
import gym_exploration
import gym_minatar

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python record.py <environment_id> <output_video_path>")
        sys.exit(1)

    env_id = sys.argv[1]
    video_path = sys.argv[2]

    print(f"Creating environment: {env_id} with render_mode='rgb_array' for recording wrapper")
    # The RecordVideo wrapper needs rgb_array output from render().
    # We will directly create the environment with render_mode='rgb_array'.
    print(f"Creating environment: {env_id} with render_mode='rgb_array' for RecordVideo wrapper") # Message updated
    try:
        env = gym.make(env_id, render_mode='rgb_array')
        print("Environment created with render_mode='rgb_array'.")
    except Exception as e_rgb: # Changed from e to e_rgb to avoid scope issues if you simplify further
        print(f"Error creating env {env_id} with render_mode='rgb_array': {e_rgb}")
        sys.exit(1)


    # Wrap with RecordVideo. This wrapper will call env.render() and expect RGB arrays.
    # It's generally robust to the env's own render_mode if env.render() returns frames.
    print(f"Wrapping environment with RecordVideo, saving to {video_path}")
    # Ensure name_prefix does not include .mp4, RecordVideo appends it.
    name_prefix_for_video = video_path.replace(".mp4", "")
    env_to_record = RecordVideo(env, video_folder=".", name_prefix=name_prefix_for_video, episode_trigger=lambda x: True)
    
    print("Resetting environment...")
    try:
        obs, info = env_to_record.reset(seed=42)
        print("Reset successful.")
    except Exception as e:
        print(f"Error during reset: {e}")
        env_to_record.close() # Close recorder
        env.close() # Close base env
        sys.exit(1)

    max_steps = 100 # Keep it short for testing
    print(f"Running for {max_steps} steps...")
    for i in range(max_steps):
        action = env_to_record.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env_to_record.step(action)
        except Exception as e:
            print(f"Error during step {i}: {e}")
            break # Exit loop on error
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps.")
            break
    else:
        print(f"Episode ran for max_steps ({max_steps}) without termination.")

    print("Closing environment and video recorder.")
    env_to_record.close() # This saves the video
    env.close() # Close the base environment
    print(f"Recording script finished. Video should be at ./{name_prefix_for_video}-episode-0.mp4 or similar.")
