# test_sapien_render.py
import gymnasium as gym
import mani_skill.envs # Essential for registering envs
import time
import traceback

print("Starting minimal SAPIEN render test...")

env = None
try:
    env = gym.make(
        "PickCube-v1", # Use a standard ManiSkill env
        # robot_uids="panda_wristcam", # Keep simple for now
        render_mode="human",
        num_envs=1
    )
    print("Environment created.")
    obs, _ = env.reset()
    print("Environment reset.")

    print("Starting render loop (close window to exit)...")
    while True:
        # Render the viewer - this is where the error occurs
        env.render_human() 
        
        # Check if viewer closed - need robust check
        if hasattr(env, 'viewer') and env.viewer and env.viewer.closed:
             print("Viewer closed by user.")
             break
             
        # Minimal simulation step (optional, can just render static scene)
        # action = env.action_space.sample() 
        # obs, reward, terminated, truncated, info = env.step(action)
        # if terminated or truncated:
        #    obs, _ = env.reset()
        
        time.sleep(0.02) # Small sleep to prevent busy-waiting

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"An error occurred: {e}")
    traceback.print_exc()
    print("-------------")

finally:
    if env is not None:
        print("Closing environment.")
        env.close()
    print("Script finished.")