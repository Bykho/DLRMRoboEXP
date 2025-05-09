import gymnasium as gym
import mani_skill.envs
import torch
import numpy as np
import sys
import os
import sapien.core as sapien # Needed for pose manipulation
from tqdm import tqdm # For potential future loops, good to have
import traceback # For printing full errors
import time # Added for sleep
import cv2 # <<< Added for visualization
import open3d as o3d # <<< Added for 3D visualization
import json # <<< ADDED for saving camera params
import plotly.graph_objs as go # <<< ADDED for Plotly visualization
from plotly.offline import plot # <<< ADDED for offline Plotly plot

# --- Attempt Pillow Import for Visualization ---
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("Warning: Pillow (PIL) not found. Scene graph PNG visualization will be disabled.")
    print("         Install it with: pip install Pillow")
# -------------------------------------------

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
    import roboexp.memory.scene_graph.graph as rsg_graph_module # For SG_index
    import roboexp.memory.robo_memory as robo_memory_module # <<< For node_label_counts
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

class InteractiveViewer:
    def __init__(self, robot_uid="panda_wristcam", control_mode="pd_ee_delta_pose", cv2_visualization=True):
        # ManiSkill Environment Setup
        self.env = None
        self.robot_uid = robot_uid
        self.control_mode = control_mode
        self.obs_mode = "image" # Request image-based observations
        self.render_mode = "human" # "human" or "rgb_array"
        self.text_prompt = (["table", "platform", "cube", "ball"], ["handle"]) # Example prompt

        self.cv2_visualization = False # SET TO FALSE
        self.display_scene_graph_image = True # Controls display of the SG png
        self.perception_run_count = 0 # <<< For unique filenames

        # RoboEXP Setup
        self.local_robo_percept = None
        self.local_robo_memory = None
        if ROBOEXP_AVAILABLE: # Use the globally set flag
            print("Initializing RoboEXP...")
            lower_bound = [-1.0, -1.0, -0.2]
            higher_bound = [1.0, 1.0, 3.5]
            voxel_size = 0.02
            object_level_labels = ["table", "platform", "cube", "ball"]
            part_level_labels = ["handle"]
            self.grounding_dict = (
                object_level_labels, # Object-level query
                part_level_labels, # Part-level query
            )
            print(f"  Grounding Dictionary: {self.grounding_dict}")
            self.output_dir = "./roboexp_output"
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"  RoboEXP output directory: {self.output_dir}")

            try:
                # Check for API key file (needed by RoboEXP's MyGPTV)
                if not os.path.exists("my_apikey"):
                     print("WARNING: 'my_apikey' file not found. RoboEXP LLM features might fail if used.")
                     print("         Creating an empty placeholder file.")
                     with open("my_apikey", "w") as f:
                          f.write("") # Write empty string or a placeholder key

                self.local_robo_memory = RoboMemory(
                    lower_bound=lower_bound,
                    higher_bound=higher_bound,
                    voxel_size=voxel_size,
                    iou_thres=0.05, # Default
                    similarity_thres=0.75, # Default
                    base_dir=self.output_dir, # <<< Use instance attribute
                )
                print("  Checking environment variables for GroundingDINO (if needed by RoboEXP)...")
                print("    GROUNDING_DINO_CONFIG_PATH:", os.getenv("GROUNDING_DINO_CONFIG_PATH"))
                print("    GROUNDING_DINO_CHECKPOINT_PATH:", os.getenv("GROUNDING_DINO_CHECKPOINT_PATH"))
                # Initialize RoboPercept passing the desired SAM type
                self.local_robo_percept = RoboPercept(
                    grounding_dict=self.grounding_dict,
                    lazy_loading=False,
                    device="cuda",
                    sam_implementation="efficientvit",
                    efficientvit_model_variant="efficientvit-sam-l0",
                    # Provide paths for GroundingDINO
                    gd_config_path="Dependencies/RoboEXP/roboexp/perception/models/config/GroundingDINO_SwinT_OGC.py",
                    gd_ckpt_path="Dependencies/RoboEXP/pretrained_models/groundingdino_swint_ogc.pth"
                )
                print("RoboEXP Initialized Successfully with EfficientViT-SAM (L0).")
            except Exception as e:
                print(f"Error initializing RoboEXP components: {e}")
                traceback.print_exc()
                # Do not change ROBOEXP_AVAILABLE here, just note init failed
                print("RoboEXP initialization failed, perception will be disabled.")
                self.local_robo_percept = None
                self.local_robo_memory = None
        # --- End RoboEXP Init ---

    # <<< ADDED 3D VISUALIZATION FUNCTION >>>
    def visualize_memory_3d(self):
        print("  Attempting to visualize RoboMemory in 3D using Plotly...")
        if not ROBOEXP_AVAILABLE or self.local_robo_memory is None:
            print("    RoboMemory not available or not initialized.")
            return

        try:
            points, colors = self.local_robo_memory.get_scene_pcd()
            if points.size == 0:
                print("    RoboMemory scene is empty. Nothing to visualize.")
                return
            
            print(f"    Visualizing {points.shape[0]} points with Plotly.")

            scatter_3d = go.Scatter3d(
                x=points[:,0], 
                y=points[:,1], 
                z=points[:,2], 
                mode='markers', 
                marker=dict(
                    size=2, # Adjusted point size for Plotly
                    color=colors, # Colors should be (N,3) RGB, 0-1 range
                    opacity=0.8
                )
            )
            
            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z'),
                    aspectmode='data' # Ensures aspect ratio is based on data range
                )
            )
            fig = go.Figure(data=[scatter_3d], layout=layout)
            
            # Plot offline (opens in a browser)
            plot_filename = os.path.join(self.output_dir, "roboexp_low_level_memory.html")
            plot(fig, filename=plot_filename, auto_open=True)
            print(f"    Plotly 3D visualization saved to and opened from: {plot_filename}")

        except Exception as e_vis3d:
            print(f"    ERROR during 3D visualization with Plotly: {e_vis3d}")
            traceback.print_exc()
        finally:
            print("    Finished 3D visualization attempt.")
    # <<< END ADDED FUNCTION >>>

    def main(self, env_id="PickCube-v1"):
        # --- Environment Configuration ---
        print(f"Initializing environment: {env_id} with robot: {self.robot_uid} and control: {self.control_mode}")

        try:
            self.env = gym.make(
                env_id,
                robot_uids=self.robot_uid,
                obs_mode="rgb+depth+segmentation+position", # <<< Correct obs_mode for textures
                control_mode=self.control_mode,
                render_mode=self.render_mode,
                num_envs=1, # Keep it single env for interactive control
                sensor_configs=dict(
                    base_camera=dict(pose=sapien.Pose(p=[0,0,0], q=[1,0,0,0]), width=128, height=128, fov=1.57, near=0.01, far=10),
                    hand_camera=dict(pose=sapien.Pose(p=[0,0,0], q=[1,0,0,0]), width=128, height=128, fov=1.57, near=0.01, far=10)
                )
            )
            # Verify action space shape after creation
            action_space_shape = self.env.action_space.shape
            print(f"Environment created. Action space: {self.env.action_space}")
            if action_space_shape != (7,):
                 print(f"WARNING: Expected action space shape (7,) for pd_ee_delta_pose, but got {action_space_shape}.")
                 if action_space_shape == (6,):
                     print("Assuming action space is (dx, dy, dz, droll, dpitch, dyaw), ignoring gripper.")
                 else:
                     print("ERROR: Unsupported action space shape for keyboard controls.")
                     self.env.close()
                     return

            # === Add a Green SPHERE to the Scene ===
            sapien_scene_obj = None
            actor_builder_obj = None
            sapien_engine_obj = None

            print("  Attempting to locate ManiSkillScene and SAPIEN Engine for adding custom actor...")
            if (hasattr(self.env, 'unwrapped') and self.env.unwrapped is not None and
                hasattr(self.env.unwrapped, 'scene') and self.env.unwrapped.scene is not None):
                
                maniskill_scene_wrapper = self.env.unwrapped.scene # This is ManiSkillScene
                print(f"    Found ManiSkillScene object: {type(maniskill_scene_wrapper)}")

                # Try to get ActorBuilder directly from ManiSkillScene
                if hasattr(maniskill_scene_wrapper, 'create_actor_builder'):
                    try:
                        actor_builder_obj = maniskill_scene_wrapper.create_actor_builder()
                        print(f"    Successfully created ActorBuilder from ManiSkillScene: {type(actor_builder_obj)}")
                    except Exception as e_builder:
                        print(f"    ERROR creating ActorBuilder from ManiSkillScene: {e_builder}")
                        actor_builder_obj = None
                else:
                    print("    WARNING: ManiSkillScene does not have 'create_actor_builder' method.")
                
                # No longer trying to explicitly find SAPIEN engine for physical material here,
                # relying on ActorBuilder to use default physical material if not specified for collision.

            else:
                print("    WARNING: self.env.unwrapped or self.env.unwrapped.scene is not available or is None. Cannot get ActorBuilder.")

            if actor_builder_obj:
                try:
                    print(f"  Attempting to add a green SPHERE to the scene using ActorBuilder: {actor_builder_obj}...")
                    builder_to_use = actor_builder_obj
                    
                    sphere_radius = 0.03
                    
                    # Add sphere collision WITHOUT explicit physical material, relying on defaults.
                    # Density is important for dynamic behavior.
                    print("    Adding sphere collision (relying on default physical material)...")
                    builder_to_use.add_sphere_collision(radius=sphere_radius, density=1000) 
                    
                    print("    Adding sphere visual...")
                    # Create a RenderMaterial for the visual appearance
                    render_material = sapien.render.RenderMaterial()
                    render_material.set_base_color([0.1, 0.8, 0.1, 1.0]) # RGBA for green
                    builder_to_use.add_sphere_visual(radius=sphere_radius, material=render_material)
                    
                    print("    Building actor...")
                    # Set the initial pose *before* building
                    table_height = 0.0
                    green_sphere_pose = sapien.Pose(p=[0.5, 0.3, table_height + sphere_radius], q=[1,0,0,0])
                    builder_to_use.initial_pose = green_sphere_pose 
                    print(f"    Set initial_pose on builder to: {green_sphere_pose.p}")
                    
                    green_sphere_actor = builder_to_use.build(name="green_sphere_obstacle")
                    
                    # Set a segmentation ID for the custom actor
                    try:
                        if hasattr(green_sphere_actor, 'set_segmentation_id'):
                            segmentation_id_for_sphere = 100 # Assign an arbitrary high ID
                            green_sphere_actor.set_segmentation_id(segmentation_id_for_sphere)
                            print(f"    Set segmentation ID for green_sphere_actor to {segmentation_id_for_sphere}")
                        elif hasattr(green_sphere_actor, 'entity') and hasattr(green_sphere_actor.entity, 'set_segmentation_id'):
                            # Sometimes need to access underlying entity
                            segmentation_id_for_sphere = 100
                            green_sphere_actor.entity.set_segmentation_id(segmentation_id_for_sphere)
                            print(f"    Set segmentation ID via green_sphere_actor.entity to {segmentation_id_for_sphere}")
                        else:
                            print("    WARNING: Could not find set_segmentation_id method on actor or actor.entity.")
                    except Exception as e_segid:
                        print(f"    ERROR setting segmentation ID: {e_segid}")

                    # Access pose via the .pose attribute for ManiSkill actors
                    print(f"  Green SPHERE added successfully at pose: p={green_sphere_actor.pose.p}, q={green_sphere_actor.pose.q}")

                except AttributeError as e_attr:
                    print(f"  ERROR: Could not add green sphere due to AttributeError: {e_attr}")
                    traceback.print_exc()
                except Exception as e_sphere:
                    print(f"  ERROR: An unexpected error occurred while trying to add green sphere: {e_sphere}")
                    traceback.print_exc()
            else:
                print("  FINAL WARNING: ActorBuilder was not created. Cannot add green sphere.")
            # === End Add Green SPHERE ===

        except Exception as e:
            print(f"Error creating environment: {e}")
            traceback.print_exc()
            if self.env: self.env.close()
            return

        # --- Print Controls ---
        print("\n" + "="*30)
        if self.render_mode == "human":
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
            print("    v:     Visualize Low-Level Memory (3D Point Cloud)")
            print("  Quit: Close the SAPIEN viewer window")
        else:
            print("Running in headless mode. Interactive control disabled.")
            print("Script will exit after basic setup check.")
            if not ROBOEXP_AVAILABLE or not self.local_robo_percept:
                 print("RoboEXP could not be initialized.")
            self.env.close()
            return
        print("="*30 + "\n")

        # --- Main Loop ---
        try:
            obs, _ = self.env.reset(seed=0)
            loop_count = 0 # For periodic pose printing

            while True:
                loop_count += 1
                # --- Periodic Pose Debugging (COMMENTED OUT) ---
                # if loop_count % 30 == 0: # Print roughly every second if loop is fast
                #     print("\n--- POSE DEBUG (Periodic) ---")
                #     try:
                #         # Robot Base Pose
                #         current_base_pose_sapien = self.env.agent.robot.get_pose()
                #         print(f"  Base Pose (World): p={current_base_pose_sapien.p}, q={current_base_pose_sapien.q}")
                # 
                #         # EE Pose
                #         ee_link_name_dbg = "panda_hand"
                #         target_link_dbg = self.env.agent.robot.find_link_by_name(ee_link_name_dbg)
                #         if target_link_dbg:
                #             current_ee_pose_sapien = target_link_dbg.pose
                #             print(f"  EE Pose ({ee_link_name_dbg}, World): p={current_ee_pose_sapien.p}, q={current_ee_pose_sapien.q}")
                #         else:
                #             print(f"  EE Pose ({ee_link_name_dbg}): Link not found.")
                #         
                #         # Camera Pose (from observations if available, might be slightly delayed)
                #         if obs and 'sensor_param' in obs and 'hand_camera' in obs['sensor_param']:
                #             cam_params_dict_dbg = obs['sensor_param']['hand_camera']
                #             if 'cam2world_gl' in cam_params_dict_dbg:
                #                 c2w_dbg = cam_params_dict_dbg['cam2world_gl']
                #                 if c2w_dbg.ndim == 3 and c2w_dbg.shape[0] == 1: c2w_dbg = c2w_dbg[0]
                #                 print(f"  HandCam c2w_gl (from obs):\n{c2w_dbg.cpu().numpy()}")
                #             else:
                #                 print("  HandCam c2w_gl: Not in current obs['sensor_param'].")
                #         else:
                #             print("  HandCam c2w_gl: No obs or sensor_param available for debug print.")
                #         print("--- END POSE DEBUG ---")
                #     except Exception as e_pose_dbg:
                #         print(f"  Error during periodic pose debug: {e_pose_dbg}")
                # --- End Periodic Pose Debugging ---

                # --- Perception Trigger (Enter key) ---
                run_perception_and_vis = False
                if ROBOEXP_AVAILABLE and self.local_robo_percept and self.local_robo_memory and hasattr(self.env, 'viewer') and self.env.viewer and not self.env.viewer.closed:
                     if self.env.viewer.window.key_press("enter"):
                          print("\n[Enter Pressed] - Running Perception and Visualizing Intermediate Output...")
                          run_perception_and_vis = True
                     if self.env.viewer.window.key_press("v"): # Keep 'v' for 3D memory vis
                         self.visualize_memory_3d()

                if run_perception_and_vis:
                    self.perception_run_count += 1
                    print(f"--- PERCEPTION & VIS RUN {self.perception_run_count} ---")
                    # <<< Clear memory for fresh perception >>>
                    if self.local_robo_memory is not None:
                        print("  Clearing RoboMemory and resetting node counters for fresh perception.")
                        self.local_robo_memory.memory_instances = []
                        self.local_robo_memory.memory_scene = {}
                        self.local_robo_memory.memory_scene_avg = {}
                        self.local_robo_memory.instance_node_mapping = {}
                        self.local_robo_memory.action_scene_graph = None # <<< Force re-creation

                        # Reset global counters used by RoboEXP
                        if ROBOEXP_AVAILABLE: # Check again to be safe
                            robo_memory_module.node_label_counts = {} # <<< Reset counter in robo_memory module
                            rsg_graph_module.SG_index = 0          # <<< Reset counter in graph module
                    
                    # <<< End Clear memory >>>
                    
                    if obs is not None:
                        print(f"  DEBUG: Top-level keys in current 'obs': {list(obs.keys())}")
                        try:
                            # === Correctly access 'hand_camera' data ===
                            sensor_data = obs.get('sensor_data')
                            sensor_params = obs.get('sensor_param')

                            if sensor_data is None or 'hand_camera' not in sensor_data:
                                print("  ERROR: 'sensor_data' or 'hand_camera' key missing in observations! Skipping.")
                                continue
                            if sensor_params is None or 'hand_camera' not in sensor_params:
                                print("  ERROR: 'sensor_param' or 'hand_camera' key missing for camera parameters! Skipping.")
                                continue

                            hand_cam_sensor_data = sensor_data['hand_camera']
                            hand_cam_sensor_params = sensor_params['hand_camera']
                            
                            # Extract tensors
                            rgb_tensor = hand_cam_sensor_data.get('rgb')
                            depth_tensor = hand_cam_sensor_data.get('depth')
                            # segmentation_tensor = hand_cam_sensor_data.get('segmentation') # Not strictly needed for RoboEXP input if not used later
                            position_tensor = hand_cam_sensor_data.get('position') # This is xyz_position for RoboEXP
                            
                            if None in [rgb_tensor, depth_tensor, position_tensor]:
                                print("  ERROR: Missing rgb, depth, or position in hand_camera sensor_data! Skipping.")
                                continue

                            # Extract camera parameters
                            cam_intrinsics_tensor = hand_cam_sensor_params.get('intrinsic_cv')
                            cam_extrinsics_gl_tensor = hand_cam_sensor_params.get('cam2world_gl')

                            if None in [cam_intrinsics_tensor, cam_extrinsics_gl_tensor]:
                                print("  ERROR: Missing intrinsic_cv or cam2world_gl in hand_camera sensor_params! Skipping.")
                                continue

                            # Convert to NumPy arrays
                            # ManiSkill often returns (1, H, W, C) or (1, H, W) for single env, so squeeze [0]
                            if rgb_tensor.ndim == 4 and rgb_tensor.shape[0] == 1: rgb_tensor = rgb_tensor[0]
                            if depth_tensor.ndim == 3 and depth_tensor.shape[0] == 1: depth_tensor = depth_tensor[0]
                            if position_tensor.ndim == 4 and position_tensor.shape[0] == 1: position_tensor = position_tensor[0]
                            # if segmentation_tensor.ndim == 3 and segmentation_tensor.shape[0] == 1: segmentation_tensor = segmentation_tensor[0]
                            if cam_intrinsics_tensor.ndim == 3 and cam_intrinsics_tensor.shape[0] == 1: cam_intrinsics_tensor = cam_intrinsics_tensor[0]
                            if cam_extrinsics_gl_tensor.ndim == 3 and cam_extrinsics_gl_tensor.shape[0] == 1: cam_extrinsics_gl_tensor = cam_extrinsics_gl_tensor[0]

                            rgb_np_uint8 = (rgb_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
                            depth_np = depth_tensor.cpu().numpy()
                            # segmentation_np = segmentation_tensor.cpu().numpy() # if used
                            position_np = position_tensor.cpu().numpy()
                            cam_intrinsics = cam_intrinsics_tensor.cpu().numpy()
                            cam_extrinsics_gl = cam_extrinsics_gl_tensor.cpu().numpy()
                            
                            observations_for_roboexp = {
                                "rgb": rgb_np_uint8,
                                "depth": depth_np, # RoboEXP might use this or rely on position
                                # "segmentation": segmentation_np, # RoboEXP's _preprocess_observation uses obs["mask"]
                                "mask": np.ones(depth_np.shape[:2], dtype=bool), # Dummy base mask for RoboEXP if it needs obs["mask"]
                                "position": position_np, # This is used by RoboEXP for point cloud creation
                                "cam_intrinsics": cam_intrinsics, # RoboEXP uses this key
                                "cam_extrinsics_gl": cam_extrinsics_gl, # RoboEXP uses this key
                                "c2w": cam_extrinsics_gl, # RoboMemory _depth_test uses this
                                "intrinsic": cam_intrinsics # RoboMemory _depth_test uses this
                            }
                            print("    Successfully prepared observations from 'hand_camera' for RoboEXP.")
                            
                            # --- Save Raw Perception Inputs (COMMENTED OUT) ---
                            # timestamp_str = time.strftime("%Y%m%d-%H%M%S")
                            # p_id = f"run{self.perception_run_count}_{timestamp_str}"
                            # cv2.imwrite(os.path.join(self.output_dir, f"{p_id}_INPUT_rgb.png"), cv2.cvtColor(rgb_np_uint8, cv2.COLOR_RGB2BGR))
                            # np.save(os.path.join(self.output_dir, f"{p_id}_INPUT_depth.npy"), depth_np)
                            # np.save(os.path.join(self.output_dir, f"{p_id}_INPUT_position.npy"), position_np)
                            # with open(os.path.join(self.output_dir, f"{p_id}_INPUT_cam_intrinsics.json"), 'w') as f: json.dump(cam_intrinsics.tolist(), f, indent=4)
                            # with open(os.path.join(self.output_dir, f"{p_id}_INPUT_cam_extrinsics_gl.json"), 'w') as f: json.dump(cam_extrinsics_gl.tolist(), f, indent=4)
                            # print(f"    Saved raw perception inputs for {p_id} to {self.output_dir}")

                            if self.local_robo_percept is not None and self.local_robo_memory is not None:
                                print("    Running RoboPercept.get_grounding_masks...")
                                # === CORRECTED CALL: Use RoboPercept's method to get masks ===
                                # This ensures that RoboPercept uses its internally processed grounding_dict string.
                                pred_boxes, pred_phrases, pred_masks_raw_sam = self.local_robo_percept.get_grounding_masks(
                                    rgb_np_uint8 # Pass the RGB image
                                    # grounding_dict is handled internally by get_grounding_masks
                                )
                                # === END CORRECTED CALL ===
                                
                                print(f"    GroundingDINO (via RoboPercept) detected: {pred_phrases}")
                                if pred_boxes is not None: print(f"    pred_boxes shape: {pred_boxes.shape}")
                                if pred_masks_raw_sam is not None: print(f"    pred_masks_raw_sam shape: {pred_masks_raw_sam.shape}")

                                # --- Save Intermediate GroundingDINO & SAM Outputs (COMMENTED OUT) ---
                                # if pred_boxes is not None: np.save(os.path.join(self.output_dir, f"{p_id}_GDINO_boxes.npy"), pred_boxes)
                                # if pred_phrases is not None:
                                #     with open(os.path.join(self.output_dir, f"{p_id}_GDINO_phrases.txt"), 'w') as f:
                                #         for phrase_idx, phrase_text in enumerate(pred_phrases):
                                #             f.write(f"Box {phrase_idx}: {phrase_text}\n")
                                # if pred_masks_raw_sam is not None: np.save(os.path.join(self.output_dir, f"{p_id}_SAM_masks.npy"), pred_masks_raw_sam)
                                # print(f"    Saved intermediate GDINO & SAM outputs for {p_id}")
                                
                                # --- Visualize and Save Detections/Masks on RGB (COMMENTED OUT) ---
                                # img_to_draw_on = rgb_np_uint8.copy()
                                # if pred_boxes is not None and pred_phrases is not None:
                                #     for i, box in enumerate(pred_boxes):
                                #         x1, y1, x2, y2 = map(int, box)
                                #         cv2.rectangle(img_to_draw_on, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                #         if i < len(pred_phrases):
                                #             cv2.putText(img_to_draw_on, pred_phrases[i], (x1, y1 - 10), 
                                #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                # if pred_masks_raw_sam is not None:
                                #     for i, mask_sam in enumerate(pred_masks_raw_sam):
                                #         if mask_sam is not None and mask_sam.ndim == 2: # Ensure mask is valid 2D
                                #             random_color = np.random.randint(50, 200, (3,), dtype=np.uint8).tolist() 
                                #             overlay = np.zeros_like(img_to_draw_on, dtype=np.uint8)
                                #             overlay[mask_sam] = random_color
                                #             cv2.addWeighted(overlay, 0.5, img_to_draw_on, 0.5, 0, img_to_draw_on)
                                # # Save the image with drawn detections/masks
                                # cv2.imwrite(os.path.join(self.output_dir, f"{p_id}_RGB_with_detections.png"), cv2.cvtColor(img_to_draw_on, cv2.COLOR_RGB2BGR))
                                # print(f"    Saved RGB with detections visualization for {p_id}")

                                # --- Display in OpenCV window (REMOVED FOR NOW) ---
                                # cv2.imshow("Perception Output (Hand Camera)", cv2.cvtColor(img_to_draw_on, cv2.COLOR_RGB2BGR))
                                # cv2.waitKey(1) # Allow window to update
                                # --- End Display ---
                                
                                # --- Construct observations_for_roboexp_attributes for RoboMemory ---
                                obs_attributes_for_memory = {
                                    'hand_camera': {
                                        'pred_phrases': pred_phrases,    # From GroundingDINO
                                        'pred_masks': pred_masks_raw_sam, # From SAM
                                        'mask_feats': None             # DenseCLIP is disabled for now
                                    }
                                }

                                # === WRAP observations_for_roboexp for RoboMemory ===
                                observations_for_memory_update = {
                                    'hand_camera': observations_for_roboexp
                                }
                                # === END WRAP ===

                                # --- Update RoboMemory and Visualize Scene Graph ---
                                self.local_robo_memory.update_memory(
                                    observations=observations_for_memory_update, # Pass the WRAPPED dict
                                    observation_attributes=obs_attributes_for_memory, # Contains Boxes, Phrases, Masks
                                    object_level_labels=self.grounding_dict[0], # Pass object_level_labels list
                                    update_scene_graph=True, # <<< ENSURE SCENE GRAPH UPDATE IS REQUESTED
                                    scene_graph_option=None # Default behavior
                                )
                                print("      RoboMemory updated.")
                                if self.local_robo_memory.action_scene_graph: # Check if graph exists
                                    self.local_robo_memory.action_scene_graph.visualize() # Call the correct method
                                    # The visualize() method in ActionSceneGraph prints its own messages and saves the file.
                                    # It does not return a path, so the following lines are removed/adjusted.
                                    print(f"      Scene graph visualization triggered from RoboMemory (see {self.output_dir}/scene_graphs/).")
                                else:
                                    print("      ActionSceneGraph not yet created in RoboMemory.")
                            else:
                                print("    local_robo_percept or RoboMemory not initialized. Skipping perception processing.")
                        except Exception as e_perc:
                            print(f"    ERROR during perception data preparation or RoboEXP call: {e_perc}")
                            traceback.print_exc()
                    else:
                        print("  Observations are None. Skipping perception.")

                # --- Keyboard Controls & Step Env (code remains the same) ---
                dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_delta = 0, 0, 0, 0, 0, 0, 0
                move_step = 0.02; rot_step = 0.05; gripper_step = 0.1
                if hasattr(self.env, 'viewer') and self.env.viewer and not self.env.viewer.closed:
                    if self.env.viewer.window.key_down('i'): dx = move_step
                    if self.env.viewer.window.key_down('k'): dx = -move_step
                    if self.env.viewer.window.key_down('j'): dy = move_step
                    if self.env.viewer.window.key_down('l'): dy = -move_step
                    if self.env.viewer.window.key_down('u'): dz = move_step
                    if self.env.viewer.window.key_down('o'): dz = -move_step
                    if self.env.viewer.window.key_down('r'): d_roll = rot_step
                    if self.env.viewer.window.key_down('f'): d_roll = -rot_step
                    if self.env.viewer.window.key_down('t'): d_pitch = rot_step
                    if self.env.viewer.window.key_down('g'): d_pitch = -rot_step
                    if self.env.viewer.window.key_down('y'): d_yaw = rot_step
                    if self.env.viewer.window.key_down('h'): d_yaw = -rot_step
                    if self.env.viewer.window.key_down('7'): gripper_delta = gripper_step
                    if self.env.viewer.window.key_down('8'): gripper_delta = -gripper_step

                action = None
                if action_space_shape == (7,):
                    action = np.array([dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_delta], dtype=np.float32)
                elif action_space_shape == (6,):
                    action = np.array([dx, dy, dz, d_roll, d_pitch, d_yaw], dtype=np.float32)
                else:
                    action = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)

                try:
                    if action is not None:
                        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.env.device).unsqueeze(0)
                        new_obs, reward, terminated, truncated, info = self.env.step(action_tensor)
                        obs = new_obs # IMPORTANT: Update obs for next loop iteration
                    else:
                        print("Error: Action was None. Skipping step.")

                    # # === ADDED/MODIFIED: Automatic Episode Reset (NOW COMMENTED OUT FOR DEMO) ===
                    # if terminated.any() or truncated.any(): # Check if any env in the batch is done
                    #     print("Episode terminated or truncated. Resetting environment.")
                    #     # Potentially log info or save final state before reset if needed
                    #     obs, info = self.env.reset(seed=0) # Reset all envs in the batch. Capture new obs and info.
                    #     print("Environment reset. New observation received.")
                    # # ============================================

                except Exception as e:
                    print(f"Error during env.step or reset: {e}")
                    traceback.print_exc()
                    # break # Decide if you want to stop on step error

                # Render the environment and handle potential closed window
                if self.env.viewer is None or not hasattr(self.env.viewer, 'window') or self.env.viewer.window is None:
                    print("Viewer window closed, exiting main loop.")
                    break
                try:
                    main_render_output = self.env.render_human()
                    # Check if the render output signals closed (some envs might do this)
                    # Add specific checks based on main_render_output if needed
                except AttributeError as e:
                    if "'NoneType' object has no attribute 'should_close'" in str(e):
                         print("Caught AttributeError related to closed window during render. Exiting loop.")
                         break
                    else:
                         print(f"Caught unexpected AttributeError: {e}")
                         raise # Re-raise other AttributeErrors
                except Exception as e:
                    print(f"An unexpected error occurred in the main loop: {e}")
                    traceback.print_exc()
                    break # Exit on other errors too

                # Optional: Add a small sleep to prevent high CPU usage if nothing happens
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Exiting.")
        except Exception as e:
             print(f"\nAn unexpected error occurred in the main loop: {e}")
             traceback.print_exc()
        finally:
            if 'env' in locals() and self.env is not None and hasattr(self.env, 'close'):
                print("Closing environment.")
                self.env.close()
            cv2.destroyAllWindows() # Close OpenCV windows on exit
            print("Script finished.")

if __name__ == "__main__":
    viewer = InteractiveViewer()
    viewer.main() # Call main on the instance 
    