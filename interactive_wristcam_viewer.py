import gymnasium as gym
import mani_skill.envs
import torch
import numpy as np
import sys
import os
import sapien.core as sapien # Needed for pose manipulation
from tqdm import tqdm # For potential future loops, good to have
import traceback # For printing full errors

# Add RoboEXP path and setup
# Make sure this path is correct relative to where you run the script
ROBOEXP_PATH = os.path.abspath("Dependencies/RoboEXP")
if ROBOEXP_PATH not in sys.path:
    sys.path.insert(0, ROBOEXP_PATH)

# === Initialize defaults before try-except block ===
ROBOEXP_AVAILABLE = False
RoboPercept = None
RoboMemory = None
# =====================================================

# --- Attempt RoboEXP Imports ---
try:
    # Check for essential dependencies first
    import open_clip
    import groundingdino
    import graphviz
    import openai # Added check

    # Now try importing the RoboEXP components
    from roboexp.perception import RoboPercept
    from roboexp.memory import RoboMemory
    ROBOEXP_AVAILABLE = True
    print("RoboEXP modules imported successfully.")
except ImportError as e:
    print("="*50)
    print(f"ERROR: Failed to import RoboEXP or its dependencies: {e}")
    print("Please ensure 'openai', 'graphviz', 'open_clip_torch', and 'groundingdino' are installed correctly.")
    print("The script will continue without RoboEXP functionality.")
    print("="*50)
    # Defaults are already set
# --- End RoboEXP Imports ---


def main():
    # --- Environment Configuration ---
    env_id = "PickCube-v1" # Keep task simple for now
    robot_uid = "panda_wristcam" # Robot with the camera
    control_mode = "pd_ee_delta_pose"
    obs_mode = "rgbd"
    render_mode = "human" # Set to None for headless
    # render_mode = None

    print(f"Initializing environment: {env_id} with robot: {robot_uid} and control: {control_mode}")

    env = None # Initialize env to None for finally block
    try:
        env = gym.make(
            env_id,
            robot_uids=robot_uid,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            num_envs=1 # Keep it single env for interactive control
        )
        # Verify action space shape after creation
        action_space_shape = env.action_space.shape
        print(f"Environment created. Action space: {env.action_space}")
        if action_space_shape != (7,):
             print(f"WARNING: Expected action space shape (7,) for pd_ee_delta_pose, but got {action_space_shape}.")
             if action_space_shape == (6,):
                 print("Assuming action space is (dx, dy, dz, droll, dpitch, dyaw), ignoring gripper.")
             else:
                 print("ERROR: Unsupported action space shape for keyboard controls.")
                 env.close()
                 return

    except Exception as e:
        print(f"Error creating environment: {e}")
        traceback.print_exc()
        if env: env.close()
        return

    # --- Initialize RoboEXP (only if imports succeeded) ---
    # These need to be local to main() or passed as arguments if needed elsewhere
    local_robo_percept = None
    local_robo_memory = None
    if ROBOEXP_AVAILABLE: # Use the globally set flag
        print("Initializing RoboEXP...")
        lower_bound = [-0.5, -0.5, -0.1]
        higher_bound = [0.5, 0.5, 0.5]
        voxel_size = 0.02
        object_level_labels = ["table", "platform", "cube"]
        part_level_labels = ["handle"]
        grounding_dict = (
            " . ".join(object_level_labels) + " . " + " . ".join(part_level_labels)
        )
        print(f"  Grounding Dictionary: {grounding_dict}")
        output_dir = "./roboexp_output"
        os.makedirs(output_dir, exist_ok=True)
        print(f"  RoboEXP output directory: {output_dir}")

        try:
            # Check for API key file (needed by RoboEXP's MyGPTV)
            if not os.path.exists("my_apikey"):
                 print("WARNING: 'my_apikey' file not found. RoboEXP LLM features might fail if used.")
                 print("         Creating an empty placeholder file.")
                 with open("my_apikey", "w") as f:
                      f.write("") # Write empty string or a placeholder key

            local_robo_memory = RoboMemory(lower_bound, higher_bound, voxel_size=voxel_size, base_dir=output_dir)
            print("  Checking environment variables for GroundingDINO (if needed by RoboEXP)...")
            print("    GROUNDING_DINO_CONFIG_PATH:", os.getenv("GROUNDING_DINO_CONFIG_PATH"))
            print("    GROUNDING_DINO_CHECKPOINT_PATH:", os.getenv("GROUNDING_DINO_CHECKPOINT_PATH"))
            # Initialize RoboPercept using the globally available class (if import succeeded)
            local_robo_percept = RoboPercept(
                grounding_dict=grounding_dict, 
                lazy_loading=False,
                sam_checkpoint_filename="sam_hq_vit_b.safetensors" # Use the smaller SAM model
            )
            print("RoboEXP Initialized Successfully.")
        except Exception as e:
            print(f"Error initializing RoboEXP components: {e}")
            traceback.print_exc()
            # Do not change ROBOEXP_AVAILABLE here, just note init failed
            print("RoboEXP initialization failed, perception will be disabled.")
            local_robo_percept = None
            local_robo_memory = None
    # --- End RoboEXP Init ---

    # --- Print Controls ---
    print("\n" + "="*30)
    if render_mode == "human":
        print("CONTROLS (SAPIEN window must be active):")
        print("  Camera: WASDQE + Mouse Drag")
        print("  Robot EE:")
        print("    i/k: Forward/Backward (+/-X)")
        print("    j/l: Left/Right      (+/-Y)")
        print("    u/o: Up/Down         (+/-Z)")
        print("    r/f: Roll            (+/-)")
        print("    t/g: Pitch           (+/-)")
        print("    y/h: Yaw             (+/-)")
        print("  Gripper:")
        print("    7:   Close Gripper")
        print("    8:   Open Gripper")
        print("  Perception:")
        print("    ENTER: Run Perception & Update Scene Graph")
        print("  Quit: Close the SAPIEN viewer window")
    else:
        print("Running in headless mode. Interactive control disabled.")
        print("Script will exit after basic setup check.")
        if not ROBOEXP_AVAILABLE or not local_robo_percept:
             print("RoboEXP could not be initialized.")
        env.close()
        return
    print("="*30 + "\n")

    # --- Main Loop ---
    try:
        obs, _ = env.reset(seed=0)

        while True:
            # --- Rendering ---
            main_render_output = env.render_human()
            if main_render_output is None and hasattr(env, 'viewer') and env.viewer and env.viewer.closed:
                 print("SAPIEN viewer closed. Exiting.")
                 break

            # --- Perception Trigger ---
            run_perception = False
            # Check if ROBOEXP was successfully *initialized*
            if ROBOEXP_AVAILABLE and local_robo_percept and local_robo_memory and hasattr(env, 'viewer') and env.viewer and not env.viewer.closed:
                 if env.viewer.window.key_press("enter"):
                      print("\n[Enter Pressed] - Running Perception and Scene Graph Update...")
                      run_perception = True

            # --- Process Perception ---
            if run_perception:
                if obs is not None:
                    print("  Capturing current observation data...")
                    try:
                        # (Data extraction code remains the same as your previous script)
                        rgb_tensor = obs['sensor_data']['hand_camera']['rgb'][0]
                        depth_tensor = obs['sensor_data']['hand_camera']['depth'][0]
                        cam_params_dict = obs['sensor_param']['hand_camera']
                        base_pose_tensor = obs['agent']['base_pose'][0]

                        # Get EE Pose (code remains the same)
                        ee_pose_world_np = None
                        base_pose_mat = np.eye(4)
                        try:
                            pos = base_pose_tensor[:3].cpu().numpy()
                            quat_wxyz = base_pose_tensor[3:].cpu().numpy()
                            s_pose = sapien.Pose(p=pos, q=quat_wxyz)
                            base_pose_mat = s_pose.to_transformation_matrix()
                            ee_link_name = env.agent.controller.configs.get("frame_id", None)
                            ee_pose_sapien = None
                            if ee_link_name:
                                for link in env.agent.robot.get_links():
                                    if link.get_name() == ee_link_name:
                                        ee_pose_sapien = link.get_pose()
                                        break
                            if ee_pose_sapien is None:
                                found_ee_link = False
                                for name_candidate in ["panda_hand", "panda_link8", env.agent.robot.get_links()[-1].get_name()]:
                                    for link in env.agent.robot.get_links():
                                        if link.get_name() == name_candidate:
                                            ee_pose_sapien = link.get_pose()
                                            found_ee_link = True
                                            break
                                    if found_ee_link: break
                            if ee_pose_sapien:
                                ee_pose_world_np = ee_pose_sapien.to_transformation_matrix()
                            else:
                                print("  Warning: Could not get live EE pose via link name...")
                                if hasattr(env.agent, 'get_ee_coords_at_base') and callable(env.agent.get_ee_coords_at_base):
                                    ee_pose_at_base_mat_tensor = env.agent.get_ee_coords_at_base()
                                    if ee_pose_at_base_mat_tensor is not None:
                                        ee_pose_at_base_mat = ee_pose_at_base_mat_tensor[0].cpu().numpy()
                                        ee_pose_world_np = base_pose_mat @ ee_pose_at_base_mat
                                    else:
                                        ee_pose_world_np = base_pose_mat
                                else:
                                     ee_pose_world_np = base_pose_mat
                        except Exception as e_pose:
                            print(f"  Error getting EE pose: {e_pose}. Using base_pose_mat as fallback.")
                            traceback.print_exc()
                            ee_pose_world_np = base_pose_mat
                        if ee_pose_world_np is None:
                            print("  FATAL: Could not determine EE pose. Using base pose.")
                            ee_pose_world_np = base_pose_mat

                        # Prepare data (code remains the same)
                        rgb_np = rgb_tensor.cpu().numpy()
                        depth_np = depth_tensor.cpu().numpy()
                        cam_params_np = {k: v[0].cpu().numpy() if hasattr(v[0], 'cpu') else v[0] for k, v in cam_params_dict.items()}
                        if depth_np.ndim == 3 and depth_np.shape[2] == 1:
                            depth_np = np.squeeze(depth_np, axis=2)
                        roboexp_obs_dict = {
                            'rgb': (rgb_np * 255).clip(0, 255).astype("uint8"),
                            'depth': depth_np,
                            'cam_param': cam_params_np,
                            'pose_world': ee_pose_world_np,
                            'pose_base': base_pose_mat
                        }
                        print(f"  Data shapes: RGB={roboexp_obs_dict['rgb'].shape}, Depth={roboexp_obs_dict['depth'].shape}")
                        print(f"  EE Pose Pos: {ee_pose_world_np[:3, 3]}")

                        # Run RoboEXP (code remains the same, using local_robo_percept/memory)
                        print("  Running RoboPercept...")
                        attributes = local_robo_percept.get_attributes_from_observations(roboexp_obs_dict)
                        # (Attribute checking and memory update code remains the same)
                        perception_keys = "None"
                        if attributes:
                            if isinstance(attributes, dict) and 'wrist' in attributes: # Check if dict and has 'wrist'
                                perception_keys = list(attributes['wrist'].keys()) if attributes['wrist'] else "Empty Wrist Attrs"
                            else: 
                                perception_keys = f"Unexpected attributes format: {type(attributes)}"
                        print(f"  RoboPercept Attributes: {perception_keys}")

                        can_update_memory = False
                        if attributes and isinstance(attributes, dict) and 'wrist' in attributes and attributes['wrist']:
                            if attributes['wrist'].get('pred_phrases') or attributes['wrist'].get('pred_instances'):
                                can_update_memory = True
                        
                        if can_update_memory:
                            print("  Updating RoboMemory Scene Graph...")
                            observations_for_memory = [obs] 
                            attributes_for_memory = [attributes] 
                                                   
                            local_robo_memory.update_memory(
                                 observations=observations_for_memory, 
                                 observation_attributes=attributes_for_memory, 
                                 object_level_labels=object_level_labels,
                                 update_scene_graph=True,
                                 scene_graph_option=None if local_robo_memory.action_scene_graph is None else {"type": "check", "old_instances": local_robo_memory.memory_instances[:]}
                            )
                            print("  Scene graph updated. Visualizing...")
                            graph_file_return = local_robo_memory.action_scene_graph.visualize() 
                            sg_filename = f"{output_dir}/sg_{local_robo_memory.action_scene_graph.SG_index-1}.gv.pdf" 
                            print(f"  Scene graph visualization likely saved to: {sg_filename}")
                            if graph_file_return: 
                                print(f"  Visualize returned: {graph_file_return}")
                        else:
                            print("  Skipping memory update due to no substantial perception attributes found for 'wrist' camera.")

                    except Exception as e:
                        print(f"  Error during RoboEXP processing: {e}")
                        traceback.print_exc()
                    print("Perception complete.")
                    run_perception = False
                else:
                     print("  Cannot run perception: No valid observation data available.")

            # --- Keyboard Controls & Step Env (code remains the same) ---
            dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_delta = 0, 0, 0, 0, 0, 0, 0
            move_step = 0.02; rot_step = 0.05; gripper_step = 0.1
            if hasattr(env, 'viewer') and env.viewer and not env.viewer.closed:
                if env.viewer.window.key_down('i'): dx = move_step
                if env.viewer.window.key_down('k'): dx = -move_step
                if env.viewer.window.key_down('j'): dy = move_step
                if env.viewer.window.key_down('l'): dy = -move_step
                if env.viewer.window.key_down('u'): dz = move_step
                if env.viewer.window.key_down('o'): dz = -move_step
                if env.viewer.window.key_down('r'): d_roll = rot_step
                if env.viewer.window.key_down('f'): d_roll = -rot_step
                if env.viewer.window.key_down('t'): d_pitch = rot_step
                if env.viewer.window.key_down('g'): d_pitch = -rot_step
                if env.viewer.window.key_down('y'): d_yaw = rot_step
                if env.viewer.window.key_down('h'): d_yaw = -rot_step
                if env.viewer.window.key_down('7'): gripper_delta = gripper_step
                if env.viewer.window.key_down('8'): gripper_delta = -gripper_step

            action = None
            if action_space_shape == (7,):
                action = np.array([dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_delta], dtype=np.float32)
            elif action_space_shape == (6,):
                action = np.array([dx, dy, dz, d_roll, d_pitch, d_yaw], dtype=np.float32)
            else:
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

            try:
                if action is not None:
                    action_tensor = torch.tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
                    new_obs, reward, terminated, truncated, info = env.step(action_tensor)
                    obs = new_obs # IMPORTANT: Update obs for next loop iteration
                else:
                    print("Error: Action was None. Skipping step.")

                # === COMMENTED OUT: Automatic Episode Reset ===
                # if terminated.any() or truncated.any(): # Check if any env in the batch is done
                #     print("Episode terminated or truncated. Resetting environment.")
                #     obs, _ = env.reset(seed=0) # Reset all envs in the batch
                # ============================================

            except Exception as e:
                print(f"Error during env.step: {e}")
                traceback.print_exc()
                # break # Decide if you want to stop on step error

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting.")
    except Exception as e:
         print(f"\nAn unexpected error occurred in the main loop: {e}")
         traceback.print_exc()
    finally:
        if 'env' in locals() and env is not None and hasattr(env, 'close'):
            print("Closing environment.")
            env.close()
        print("Script finished.")

if __name__ == "__main__":
    main() 