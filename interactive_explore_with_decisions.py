# This script is likely named interactive_explore.py

from roboexp import (
    RobotExploration,
    RoboMemory,
    RoboPercept,
    RoboAct,
    RoboDecision,
)
from datetime import datetime
import os
import json
import sapien.core as sapien
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
import numpy as np
import cv2

# Import PyTorch and functional for memory management and resizing
import torch
import torch.nn.functional as F

# Import the utility function for camera positioning (already present)
from roboexp.utils import get_pose_look_at

# ----------------------------------------------------------------------
# Environment Variables for Headless Rendering with NVIDIA GPU
os.environ["SAPIEN_HEADLESS"] = "1"

# Try multiple possible Vulkan ICD paths
vulkan_paths = [
    "/etc/vulkan/icd.d/nvidia_icd.json",
    "/usr/share/vulkan/icd.d/nvidia_icd.json",
    # Add any other potential paths your system might use
]

# Find the first existing Vulkan ICD path
vulkan_path = None
for path in vulkan_paths:
    if os.path.exists(path):
        vulkan_path = path
        break

if vulkan_path:
    os.environ["VK_ICD_FILENAMES"] = vulkan_path
    print(f"Found Vulkan ICD at: {vulkan_path}")
else:
    print("Warning: No Vulkan ICD found in standard locations")

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

# Make sure we're using the virtual display
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":99"

# --- Add memory clearing at the very beginning ---
# This helps ensure a clean start with maximum free GPU memory
torch.cuda.empty_cache()
print("✨ GPU memory cache cleared at start")
# -------------------------------------------------


def save_frame(frame, frame_dir, frame_num):
    """Save a frame as an image."""
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    frame_path = os.path.join(frame_dir, f"frame_{frame_num:06d}.png")

    # --- CORRECTED: Convert from float [0, 1] to uint8 [0, 255] ---
    # Assuming the input 'frame' (rgb_resized_np) is float in range [0, 1]
    frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)
    # -------------------------------------------------------------

    # Convert from RGB (uint8) to BGR (uint8) for OpenCV
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(frame_path, frame_bgr)



def explore(robo_decision, robo_act, frame_dir, wrist_cam):
    """Test exploration driven by RoboDecision"""

    frame_num = 0
    PERCEPTION_W, PERCEPTION_H = 256, 256
    W_orig, H_orig = 512, 512

    if frame_dir and not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
        print(f"Created frame directory: {frame_dir}")

    # Camera examination lines before loop (uncommented)
    print("DEBUG: Examining camera structure from robo_act.robo_exp")
    if hasattr(robo_act, 'robo_exp') and hasattr(robo_act.robo_exp, 'cameras'):
         print(f"DEBUG: Available camera keys: {list(robo_act.robo_exp.cameras.keys())}")
    else:
         print("DEBUG: robo_act.robo_exp or cameras attribute not available.")


    # --- Overall Exploration Loop ---
    overall_step = 0
    max_overall_steps = 5

    # You might need an initial pose or strategy to get the first observation
    # For simplicity, let's start at the default robot init pose and capture from there
    # initial_pose = robo_act.robo_exp.robot.get_qpos()


    overall_exploration_finished = False

    while overall_step < max_overall_steps and not overall_exploration_finished:
        overall_step += 1
        print(f"\n--- Overall Exploration Step {overall_step} ---")

        print("Attempting to capture observation and update memory...")
        # Use a more specific try...except for better error reporting
        try:
            # --- Capture, Perception, Memory Update Block ---
            try: # Inner try block for capture/perception/memory steps
                # Force scene update
                robo_act.robo_exp.scene.update_render()

                # Use the explicitly created wrist_cam object
                camera = wrist_cam

                # Capture and Get Textures
                print(f"   GPU Memory before capture: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                camera.take_picture()
                print("   📸 Camera picture taken.")

                available_textures = []
                if hasattr(camera, 'get_available_texture_names'):
                    available_textures = camera.get_available_texture_names()
                print(f"   Available textures: {available_textures}")

                rgb_texture_obj = None
                pos4_texture_obj = None
                try:
                    rgb_texture_obj = camera.get_float_texture("Color")
                    pos4_texture_obj = camera.get_float_texture("Position")
                    print(f"   get_float_texture('Color') returned: {type(rgb_texture_obj)}")
                    print(f"   get_float_texture('Position') returned: {type(pos4_texture_obj)}")
                except Exception as e:
                    print(f"   Error getting textures: {e}")
                    raise # Re-raise to be caught by the outer except

                if rgb_texture_obj is None or pos4_texture_obj is None:
                    print("   ❌ Required textures not available. Skipping this step.")
                    raise ValueError("Required textures not available") # Re-raise with message

                rgb_orig   = rgb_texture_obj[..., :3]
                pos4_orig  = pos4_texture_obj

                print(f"   RGB original shape: {rgb_orig.shape}")
                print(f"   RGB original min: {np.min(rgb_orig)}, max: {np.max(rgb_orig)}, mean: {np.mean(rgb_orig)}")
                # print(f"   Depth original min: {np.min(pos4_orig[:,:,2])}, max: {np.max(pos4_orig[:,:,2])}, mean: {np.mean(pos4_orig[:,:,2])}")


                # Resize Images and Create Obs Dictionary
                rgb_torch = torch.from_numpy(rgb_orig).permute(2, 0, 1).unsqueeze(0).cuda()
                pos4_torch = torch.from_numpy(pos4_orig).permute(2, 0, 1).unsqueeze(0).cuda()

                rgb_resized = F.interpolate(rgb_torch, size=(PERCEPTION_H, PERCEPTION_W), mode='bilinear', align_corners=False)
                pos4_resized = F.interpolate(pos4_torch, size=(PERCEPTION_H, PERCEPTION_W), mode='nearest')

                rgb_resized_np = rgb_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
                pos4_resized_np = pos4_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()

                print(f"   RGB resized shape: {rgb_resized_np.shape}")
                print(f"   RGB resized min: {np.min(rgb_resized_np)}, max: {np.max(rgb_resized_np)}, mean: {np.mean(rgb_resized_np)}")
                # print(f"   Depth resized min: {np.min(pos4_resized_np[:,:,2])}, max: {np.max(pos4_resized_np[:,:,2])}, mean: {np.mean(pos4_resized_np[:,:,2])}")

                obs = dict(
                    rgb       = rgb_resized_np,
                    position  = pos4_resized_np[..., :3],
                    mask      = pos4_resized_np[..., 3] < 1,
                    c2w       = camera.get_model_matrix(),
                    intrinsic = camera.get_intrinsic_matrix(),
                    depths     = pos4_resized_np[..., 2],
                )

                # --- Call Perception ---
                print(f"   GPU Memory before perception: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                if not hasattr(robo_act, 'robo_percept') or robo_act.robo_percept is None:
                     print("   DEBUG: RoboPercept module not available on robo_act. Skipping perception.")
                     # Re-raise if perception is critical
                     # raise AttributeError("RoboPercept module not available")
                     attrs = {} # Provide empty attrs if skipping
                else:
                    attrs = robo_act.robo_percept.get_attributes_from_observations({"wrist": obs})
                    print("   🔍 percept keys:", list(attrs.keys()))
                    # --- DEBUG PRINT: Inspect Perception Output ---
                    print(f"   DEBUG: Perception attributes (first 5): {dict(list(attrs.items())[:5])}")
                    # You might want to inspect the structure of attrs further, e.g., attrs['wrist']['instances']
                    # if 'wrist' in attrs and 'instances' in attrs['wrist']:
                    #      print(f"   DEBUG: Detected instances count: {len(attrs['wrist']['instances'])}")
                    #      if len(attrs['wrist']['instances']) > 0:
                    #           print(f"   DEBUG: First detected instance label: {attrs['wrist']['instances'][0]['label']}")
                    # ---------------------------------------------
                    print(f"   GPU Memory after perception: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


                # --- Call Memory Update ---
                if not hasattr(robo_act, 'robo_memory') or robo_act.robo_memory is None:
                     print("   DEBUG: RoboMemory module not available on robo_act. Skipping memory update.")
                     # Re-raise if memory is critical
                     # raise AttributeError("RoboMemory module not available")
                     pass # Continue without memory update if not available
                else:
                    robo_act.robo_memory.update_memory(
                        observations           = {"wrist": obs},
                        observation_attributes = attrs,
                        object_level_labels    = ["table"], # Ensure this list is correct for your scene config
                        filter_masks           = {},
                        update_scene_graph=True,
                        scene_graph_option=None,
                    )
                    print("   💾 memory updated")
                    # --- DEBUG PRINT: Inspect Memory Contents ---
                    print(f"   DEBUG: Memory instances count: {len(robo_act.robo_memory.memory_instances)}")
                    print(f"   DEBUG: Scene graph object nodes count: {len(robo_act.robo_memory.action_scene_graph.object_nodes) if robo_act.robo_memory.action_scene_graph else 0}")
                    if robo_act.robo_memory.action_scene_graph and robo_act.robo_memory.action_scene_graph.object_nodes:
                         print(f"   DEBUG: Scene graph object nodes: {list(robo_act.robo_memory.action_scene_graph.object_nodes.keys())}")
                    # --------------------------------------------
                    print(f"   GPU Memory after memory update: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


                # Save the frame (using the resized RGB)
                save_frame(rgb_resized_np, frame_dir, frame_num)
                print(f"Saved frame to {frame_dir}/frame_{frame_num:06d}.png")
                # frame_num += 1 # Increment after a full action execution cycle, or per saved frame

            except Exception as e: # Catch specific errors within the observation/memory block
                print(f"   ❌ Error during capture, perception, or memory update step {overall_step}: {type(e).__name__}: {e}")
                # Decide how to handle errors - maybe continue, maybe break
                continue # Continue to the next overall step if observation/memory fails

            finally: # Memory freeing
                if 'obs' in locals(): del obs
                if 'attrs' in locals(): del attrs
                if 'rgb_torch' in locals(): del rgb_torch
                if 'pos4_torch' in locals(): del pos4_torch
                if 'rgb_resized' in locals(): del rgb_resized
                if 'pos4_resized' in locals(): del pos4_resized
                if 'rgb_orig' in locals(): del rgb_orig
                if 'pos4_orig' in locals(): del pos4_orig
                if 'rgb_texture_obj' in locals(): del rgb_texture_obj
                if 'pos4_texture_obj' in locals(): del pos4_texture_obj

                torch.cuda.empty_cache()
                print("   ✨ GPU memory freed after observation/memory update")
                print(f"   GPU Memory after clearing cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"   Max GPU Memory allocated so far (peak): {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            # --- End of Capture, Perception, Memory Update Block ---


            # --- NEW: Update Decision's Action List ---
            print("Updating decision's action list based on memory...")
            try:
                # This populates robo_decision.action_list based on the updated memory
                robo_decision.update_action_list()
                print(f"   Decision action list size after update: {len(robo_decision.action_list)}")
            except Exception as e:
                # Modified error print to show type and message
                print(f"   ❌ Error updating decision action list: {type(e).__name__}: {e}")
                # Decide how to handle - maybe continue to next overall step, maybe break
                continue # Continue to the next overall step

            # --- NEW: Execute Actions Decided by RoboDecision ---
            print("Executing actions from decision list...")
            executed_action_in_step = False # Flag to know if we executed anything

            # Loop while there are actions in the decision list AND we haven't executed one yet
            while not robo_decision.is_done() and not executed_action_in_step:
                try:
                    action_tuple = robo_decision.get_action()
                    node, action_type = action_tuple
                    print(f"   -> Decided action: '{action_type}' on node: '{node.node_id}'")

                    success = False

                    if action_type == "open_close":
                        print(f"      Attempting to execute skill_open_close on {node.node_id}")
                        try:
                             robo_act.skill_open_close(node, visualize=False)
                             success = True
                        except Exception as skill_e:
                             print(f"      ❌ Error executing skill_open_close: {type(skill_e).__name__}: {skill_e}")
                             success = False

                    elif action_type == "pick_away":
                        print(f"      Attempting to execute placeholder pick_away on {node.node_id}")
                        # Placeholder - replace with actual pick_away logic
                        print("         (Placeholder execution for pick_away)")
                        success = True


                    elif action_type == "pick_back":
                        print(f"      Attempting to execute placeholder pick_back on {node.node_id}")
                        # Placeholder - replace with actual pick_back logic
                        print("         (Placeholder execution for pick_back)")
                        success = True

                    elif action_type == "no_action":
                        print(f"      Decided: No specific action needed for node {node.node_id} at this time.")
                        success = True

                    else:
                        print(f"      ⚠️ Warning: Unknown action type from decision: '{action_type}'. Skipping execution.")
                        success = False

                    # --- Update Decision Memory Based on Action Outcome (Optional) ---
                    # If RoboDecision has a method to update its state after an action:
                    # try:
                    #      robo_decision.update_decision_memory(executed_action=action_tuple, success=success)
                    # except Exception as update_e:
                    #      print(f"      ❌ Error updating decision memory: {type(update_e).__name__}: {update_e}")


                    executed_action_in_step = True # Mark that we attempted an action in this overall step

                except Exception as e:
                     # Modified error print
                     print(f"   ❌ Error getting or executing action from decision: {type(e).__name__}: {e}")
                     executed_action_in_step = True # Treat error as having attempted the action for this step
                     # Decide if a critical error here should stop exploration
                     # overall_exploration_finished = True # Example: stop on execution error
                     continue # Skip to next overall step

            # After executing one action (or attempting to), we go back to the start of the
            # outer loop to observe the potentially changed scene.
            # If no actions were generated or executed in the inner loop, the outer loop
            # will still proceed to the next observation step (up to max_overall_steps).

            # Increment frame_num and save frame here, once per overall exploration step
            # This assumes each overall step potentially yields a new observation frame to save
            frame_num += 1

        except Exception as outer_e:
            print(f"❌ Error in overall exploration step {overall_step}: {type(outer_e).__name__}: {outer_e}")
            continue

    print("\nExploration sequence completed")
    print(f"Finished after {overall_step} overall steps.")

    # Optional: Visualize the final scene graph
    if hasattr(robo_decision.robo_memory, 'action_scene_graph') and robo_decision.robo_memory.action_scene_graph:
        try:
            print("Visualizing the final scene graph")
            robo_decision.robo_memory.action_scene_graph.visualize()
        except Exception as viz_e:
            print(f"Error visualizing scene graph: {type(viz_e).__name__}: {viz_e}")


# Keep the run function as it was in the previous turn (all module initializations uncommented)
def run(base_dir, REPLAY_FLAG=False):
    # Load simulation configuration
    with open("config/simulation_config.json", "r") as f:
        sim_config = json.load(f)

    # --- ENSURE THIS BLOCK IS FULLY ACTIVE ---
    simplified_objects = []
    for obj in sim_config["objects_conf"]:
        # Create a new object config with split pose
        new_obj = obj.copy()
        if "init_pose" in obj:
            # Split init_pose into position and rotation
            new_obj["init_pos"] = obj["init_pose"][:3]
            new_obj["init_rot"] = obj["init_pose"][3:]
            new_obj.pop("init_pose", None)
        simplified_objects.append(new_obj)
    # --- END BLOCK ---

    print(f"DEBUG: objects_conf being passed to RobotExploration: {simplified_objects}") # Keep this debug print
    print(f"DEBUG: Type of objects_conf being passed: {type(simplified_objects)}") # Keep this debug print
    if simplified_objects: # Keep this debug print
        print(f"DEBUG: Type of first object dict: {type(simplified_objects[0])}")
        if simplified_objects[0]:
            print(f"DEBUG: Keys in first object dict: {simplified_objects[0].keys()}")


    #print(f"DEBUG: Using simplified object configuration with {len(simplified_objects)} objects")

    # *** Set Ray tracing render config BEFORE RobotExploration ***
    print("DEBUG: Setting Ray Tracing render config...")
    sapien.render_config.camera_shader_dir    = "rt"
    sapien.render_config.viewer_shader_dir    = "rt"
    sapien.render_config.rt_samples_per_pixel = 4
    sapien.render_config.rt_use_denoiser      = False
    print("DEBUG: Ray Tracing render config set.")
    # -----------------------------------------------------------

    # Initialize the robot exploration with minimal settings
    # This is where the renderer is initialized based on config
    print("DEBUG: Initializing RobotExploration...")
    robo_exp = RobotExploration(
        data_path=sim_config["data_path"],
        robot_conf=sim_config["robot_conf"],
        objects_conf=simplified_objects,  # <--- Make sure this passes simplified_objects
        ray_tracing=True,
        balance_passive_force=True,
        offscreen_only=True,
        gt_depth=False,
        has_gripper=False,
        control_mode="mplib",
    )
    print("DEBUG: RobotExploration initialized.")

    # --- AGGRESSIVE LIGHTING SIMPLIFICATION BLOCK ---
    print("DEBUG: Clearing all existing lights and adding simple ones...")
    try:
        lights = list(robo_exp.scene.get_all_lights())
        print(f"DEBUG: Found {len(lights)} existing lights. Attempting to remove them.")
        for light in lights:
            robo_exp.scene.remove_light(light)
        print("DEBUG: Existing lights removed.")
    except Exception as e:
        print(f"Warning: Could not remove existing lights: {e}")

    print("DEBUG: Simplifying scene lighting - Adding ambient light")
    robo_exp.scene.set_ambient_light([0.3, 0.3, 0.3])

    print("DEBUG: Simplifying scene lighting - Adding directional light")
    robo_exp.scene.add_directional_light(direction=[1,-1,-1], color=[1,1,1], shadow=False) # No shadows for simplicity
    # --------------------------------------------------------------


    # --- Add explicit camera creation ---
    print("DEBUG: Adding explicit camera 'wrist_cam'...")
    end_link  = robo_exp.robot.get_links()[-1]
    cam_pose  = sapien.Pose([0, 0, 0.10], [1, 0, 0, 0])
    W, H      = 512, 512

    wrist_offset = sapien.Pose([0, 0, 0.1], [1, 0, 0, 0])
    wrist_cam = robo_exp.scene.add_mounted_camera(
        "wrist_cam", end_link, wrist_offset,
        width=W, height=H, fovy=np.deg2rad(60), near=0.1, far=100,
    )
    print(f"DEBUG: Camera 'wrist_cam' added. (Note: Explore uses 'wrist' camera from robot config)")
    # --------------------------------------


    # --- INITIALIZE ALL ROBOEXP MODULES (UNCOMMENTED) ---
    print("DEBUG: Initializing Memory module...")
    robo_memory = RoboMemory(
         lower_bound=[0, -0.8, -1],
         higher_bound=[1, 0.5, 2],
         voxel_size=0.01,
         real_camera=True,
         base_dir=base_dir,
         similarity_thres=0.95,
         iou_thres=0.01,
    )
    print("DEBUG: Memory module initialized.")

    print("DEBUG: Initializing Perception module...")
    object_level_labels = ["table", "cabinet", "glasses"]
    part_level_labels = []
    grounding_dict = (
         " . ".join(object_level_labels) + " . " + " . ".join(part_level_labels)
    )
    robo_percept = RoboPercept(grounding_dict=grounding_dict, lazy_loading=True, device="cuda", use_sam_hq_in_segment=False)
    print("DEBUG: Perception module initialized.")

    print("DEBUG: Initializing Action module...")
    robo_act = RoboAct(
        robo_exp, # Pass robo_exp
        robo_percept, # Pass robo_percept
        robo_memory,  # Pass robo_memory
        object_level_labels,
        base_dir=base_dir,
        REPLAY_FLAG=REPLAY_FLAG,
    )
    print("DEBUG: Action module initialized.")

    print("DEBUG: Initializing Decision module...")
    robo_decision = RoboDecision(robo_memory, base_dir, REPLAY_FLAG=REPLAY_FLAG)
    print("DEBUG: Decision module initialized.")
    # ------------------------------------------------------------------------


    # Create frames directory
    frame_dir = os.path.join(base_dir, "frames")


    # Call explore with the initialized modules
    explore(robo_decision, robo_act, frame_dir, wrist_cam)

    print("\nDEBUG: run function finished initialization phase.")

# Main execution block (already present)
if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{current_time}"
    REPLAY_FLAG = False
    if not os.path.exists(base_dir):
        # Create directory if it doesn't exist
        os.makedirs(base_dir)
    run(base_dir, REPLAY_FLAG=REPLAY_FLAG)
