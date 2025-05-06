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



def explore(robo_decision, robo_act, frame_dir, wrist_cam): # Updated signature
    """Test exploration with saved observations"""

    # Modules are expected to be initialized in run now

    frame_num = 0
    PERCEPTION_W, PERCEPTION_H = 256, 256
    W_orig, H_orig = 512, 512

    if frame_dir and not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
        print(f"Created frame directory: {frame_dir}")

    test_poses = [
        [0.2, -0.1, 0.15, 1.0, 0.0, 0.0, 0.0],  # left
        [0.2,  0.0, 0.15, 1.0, 0.0, 0.0, 0.0],  # center
        [0.2,  0.1, 0.15, 1.0, 0.0, 0.0, 0.0],  # right
    ]

    # --- DEBUG STEP: Uncommented camera examination ---
    # This needs robo_act.robo_exp to be initialized, which it is
    print("DEBUG: Examining camera structure from robo_act.robo_exp")
    if hasattr(robo_act, 'robo_exp') and hasattr(robo_act.robo_exp, 'cameras'):
         # This line will still print keys from robo_act.robo_exp.cameras, which is fine for info
         print(f"DEBUG: Available camera keys: {list(robo_act.robo_exp.cameras.keys())}")
    else:
         # This will only happen if robo_act wasn't initialized, which is not the case now.
         print("DEBUG: robo_act.robo_exp or cameras attribute not available.")
    # ---------------------------------------------------------


    for step, pose in enumerate(test_poses, 1):
        print(f"\n[{step}/{len(test_poses)}] Testing pose: {pose}")
        try: # Outer try block for movement errors (uncommented)
            # Validate position
            position = pose[:3]
            distance = np.linalg.norm(position)
            if distance > 0.35:
                print(f"Skipping pose {pose} - likely outside workspace (distance={distance:.2f}m)")
                continue

            print(f"Attempting movement to: {pose}")
            # Corrected: Call run_action on robo_act.robo_exp
            success = robo_act.robo_exp.run_action(
                action_code=1,
                action_parameters=pose,
                iteration=1000
            )

            if not success:
                print("Movement failed, skipping observation")
                continue

            print("Movement complete, attempting to capture observation...")

            # Force scene update (now uncommented)
            robo_act.robo_exp.scene.update_render()

            # --- DEBUG STEP: Uncommented Get Camera Object Block AND take_picture() ---
            try: # Inner try block for capture/perception/memory

                # --- Directly use the wrist_cam object passed to the function ---
                # Note: We are NOT getting the camera from robo_act.robo_exp.cameras["wrist"] here
                camera = wrist_cam # Use the explicitly created camera object

                # --- CONTINUE UNCOMMENTING FROM HERE ---
                # These lines were the cause of segfault previously.
                print(f"   GPU Memory before capture: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                # Try to take a picture
                camera.take_picture() # <--- Testing if segfault happens HERE using the explicit camera
                print("   📸 Camera picture taken.")

                # Add debug prints for available textures
                available_textures = []
                if hasattr(camera, 'get_available_texture_names'):
                    available_textures = camera.get_available_texture_names()
                print(f"   Available textures: {available_textures}")

                # Try to get the image data (Need to uncomment this block next if take_picture works)
                rgb_texture_obj = None
                pos4_texture_obj = None # Also check for Position
                try:
                    rgb_texture_obj = camera.get_float_texture("Color")
                    print(f"   get_float_texture('Color') returned: {type(rgb_texture_obj)}")
                except Exception as e:
                    print(f"   Error getting 'Color' texture: {e}")
                
                try:
                    pos4_texture_obj = camera.get_float_texture("Position")
                    print(f"   get_float_texture('Position') returned: {type(pos4_texture_obj)}")
                except Exception as e:
                     print(f"   Error getting 'Position' texture: {e}")
                
                if rgb_texture_obj is None or pos4_texture_obj is None:
                    print("   ❌ Required textures not available. Skipping this step.")
                    continue # Skip if either texture is missing
                
                rgb_orig   = rgb_texture_obj[..., :3] # Get RGB channels
                pos4_orig  = pos4_texture_obj # Get Position (XYZW)

                # --- DEBUG STEP: Add prints for raw image data ---
                print(f"   RGB original shape: {rgb_orig.shape}")
                print(f"   RGB original min: {np.min(rgb_orig)}, max: {np.max(rgb_orig)}, mean: {np.mean(rgb_orig)}")
                print(f"   Depth original min: {np.min(pos4_orig[:,:,2])}, max: {np.max(pos4_orig[:,:,2])}, mean: {np.mean(pos4_orig[:,:,2])}")
                # --------------------------------------------------

                # Convert numpy to torch tensor and move to CUDA (HWC -> NCHW for F.interpolate)
                rgb_torch = torch.from_numpy(rgb_orig).permute(2, 0, 1).unsqueeze(0).cuda()
                pos4_torch = torch.from_numpy(pos4_orig).permute(2, 0, 1).unsqueeze(0).cuda()

                # Resize RGB using bilinear interpolation
                rgb_resized = F.interpolate(rgb_torch, size=(PERCEPTION_H, PERCEPTION_W), mode='bilinear', align_corners=False)

                # Resize Position/Mask/Depth using nearest neighbor interpolation
                pos4_resized = F.interpolate(pos4_torch, size=(PERCEPTION_H, PERCEPTION_W), mode='nearest')

                # Convert resized tensors back to numpy and to CPU
                rgb_resized_np = rgb_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
                pos4_resized_np = pos4_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()

                # --- DEBUG STEP: Add prints for resized image data ---
                print(f"   RGB resized shape: {rgb_resized_np.shape}")
                print(f"   RGB resized min: {np.min(rgb_resized_np)}, max: {np.max(rgb_resized_np)}, mean: {np.mean(rgb_resized_np)}")
                print(f"   Depth resized min: {np.min(pos4_resized_np[:,:,2])}, max: {np.max(pos4_resized_np[:,:,2])}, mean: {np.mean(pos4_resized_np[:,:,2])}")
                # ----------------------------------------------------

                # Create obs dictionary using the RESIZED data
                obs = dict(
                    rgb       = rgb_resized_np,              # Use resized RGB
                    position  = pos4_resized_np[..., :3],    # Use resized position
                    mask      = pos4_resized_np[..., 3] < 1, # Derive mask from resized pos4
                    c2w       = camera.get_model_matrix(),   # Still original camera matrix
                    intrinsic = camera.get_intrinsic_matrix(), # Still original camera matrix
                    # note: The intrinsic matrix is for the ORIGINAL camera resolution (512x512).
                #     # If your perception module uses the intrinsic matrix with the image data,
                #     # you might need to adjust the intrinsic matrix based on the scaling factor (PERCEPTION_W/W_orig, PERCEPTION_H/H_orig),
                #     # or ensure the perception module handles input images at different resolutions correctly.
                    depths     = pos4_resized_np[..., 2], # Derive depth from resized pos4
                )
                # ---------------------------------------------


                # --- Call Perception and Memory Update (Need to uncomment this block after creating obs dict) ---
                # # Add memory profiling print
                print(f"   GPU Memory before perception: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                #
                # # perception → attributes
                # # Requires robo_act.robo_percept to be initialized
                if not hasattr(robo_act, 'robo_percept') or robo_act.robo_percept is None:
                    print("   DEBUG: RoboPercept module not available on robo_act. Skipping perception.")
                #      # Still save the frame if possible, even without full perception
                    if 'rgb_resized_np' in locals() and rgb_resized_np is not None:
                        save_frame(rgb_resized_np, frame_dir, frame_num)
                        print(f"Saved raw frame to {frame_dir}/frame_{frame_num:06d}.png")
                        frame_num += 1
                    continue # Skip the rest of the try block
                #
                attrs = robo_act.robo_percept.get_attributes_from_observations({"wrist": obs})
                print("   🔍 percept keys:", list(attrs.keys()))
                print(f"   GPU Memory after perception: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                #
                # # memory update
                # # Requires robo_act.robo_memory to be initialized
                if not hasattr(robo_act, 'robo_memory') or robo_act.robo_memory is None:
                    print("   DEBUG: RoboMemory module not available on robo_act. Skipping memory update.")
                    # Still save the frame if possible, even without full perception
                    if 'rgb_resized_np' in locals() and rgb_resized_np is not None:
                        save_frame(rgb_resized_np, frame_dir, frame_num)
                        print(f"Saved raw frame to {frame_dir}/frame_{frame_num:06d}.png")
                        frame_num += 1
                    continue # Skip the rest of the try block
                #
                #
                robo_act.robo_memory.update_memory(
                    observations           = {"wrist": obs}, # Pass the 'obs' dictionary
                    observation_attributes = attrs,
                    object_level_labels    = ["table"], # Use the same labels as defined in run
                    filter_masks           = {},     # never None
                )
                print("   💾 memory updated")
                print(f"   GPU Memory after memory update: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                #
                # # Save the frame (using the resized RGB)
                save_frame(rgb_resized_np, frame_dir, frame_num)
                print(f"Saved frame to {frame_dir}/frame_{frame_num:06d}.png")
                frame_num += 1
                # -----------------------------------------


            except Exception as e:
                # This block catches errors during capture, resizing, perception, or memory update
                print(f"   ❌ Error during perception/memory update for step {step}: {e}")
                # Use a broad except Exception to catch OOM and other issues
                # Continue to the next pose even if one fails
                continue # Skip to the next pose

            finally:
                # --- Memory freeing ---
                # Ensure memory is freed even if an error occurs in try block
                # Check if variables exist before deleting - uncomment del if the variable
                # is created in the *uncommented* code in the try block.
                if 'obs' in locals(): del obs # Keep commented
                if 'attrs' in locals(): del attrs # Keep commented
                if 'rgb_torch' in locals(): del rgb_torch # Keep commented
                if 'pos4_torch' in locals(): del pos4_torch # Keep commented
                if 'rgb_resized' in locals(): del rgb_resized # Keep commented
                if 'pos4_resized' in locals(): del pos4_resized # Keep commented
                # These were numpy arrays from getting textures - Uncomment if texture getting is uncommented
                if 'rgb_orig' in locals(): del rgb_orig # Keep commented
                if 'pos4_orig' in locals(): del pos4_orig # Keep commented
                # These were objects returned by get_float_texture - Uncomment if texture getting is uncommented
                if 'rgb_texture_obj' in locals(): del rgb_texture_obj # Keep commented
                if 'pos4_texture_obj' in locals(): del pos4_texture_obj # Keep commented


                torch.cuda.empty_cache() # Keep uncommented - always good to clear cache
                print("   ✨ GPU memory freed")
                print(f"   GPU Memory after clearing cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"   Max GPU Memory allocated so far (peak): {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
                # ----------------------


        except Exception as e: # Outer except for movement errors
            print(f"Error during pose testing (movement or initial capture setup): {str(e)}")
            # This catch is less likely now that movement is working
            continue # Continue outer loop to next pose

    print("\nExploration sequence completed")



# Keep the run function as it was in the previous turn (all module initializations uncommented)
def run(base_dir, REPLAY_FLAG=False):
    # Load simulation configuration
    with open("config/simulation_config.json", "r") as f:
        sim_config = json.load(f)

    # Create a simplified objects list - just keep the table
    simplified_objects = []
    for obj in sim_config["objects_conf"]:
        if "name" in obj and obj["name"] == "table":
            simplified_objects.append(obj)
            break  # Just one object

    print(f"DEBUG: Using simplified object configuration with {len(simplified_objects)} objects")

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
        objects_conf=simplified_objects,  # Only the essential objects
        ray_tracing=True,  # Keep ray tracing enabled
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
    object_level_labels = ["table"]
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