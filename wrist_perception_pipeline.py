# wrist_perception_pipeline.py
import os
import matplotlib.pyplot as plt
import sys
import mplib
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Import SAPIEN components
import sapien.core as sapien
from roboexp import RoboPercept, RoboMemory
from roboexp.utils import get_pose_look_at

# Clear GPU memory at start
torch.cuda.empty_cache()
print("✨ GPU memory cache cleared at start")

# Setup repository paths
repo_root = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(repo_root, "pipeline_output"), exist_ok=True)

# helper ---------------------------------------------------------------
def point_wrist_cam_at(target_pose, link, cam, offset=np.array([0, 0, 0.10])):
    """
    Re‑orient the existing wrist_cam so its –Z axis points at `target_pose`.
    The camera origin is the link frame translated by `offset` (in link local).
    """
    link_pose = link.get_pose()                      # link → world
    R = link_pose.to_transformation_matrix()[:3, :3] # rotation 3×3
    cam_origin_world = link_pose.p + R @ offset      # manual transform

    T_wc = get_pose_look_at(cam_origin_world, target_pose.p)
    cam_pose_world = sapien.Pose.from_transformation_matrix(T_wc)

    cam.set_local_pose(link_pose.inv() * cam_pose_world)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 1) Setup engine & scene
print("Setting up engine and scene...")
engine = sapien.Engine()
renderer = sapien.SapienRenderer(offscreen_only=True)
engine.set_renderer(renderer)

# Configure ray tracing settings
sapien.render_config.camera_shader_dir = "rt"
sapien.render_config.viewer_shader_dir = "rt"
sapien.render_config.rt_samples_per_pixel = 1  # Reduced for stability
sapien.render_config.rt_use_denoiser = False
sapien.render_config.rt_max_bounces = 1
sapien.render_config.rt_shadows_enabled = False
sapien.render_config.rt_light_source_emission_samples = 1

scene = engine.create_scene()
scene.set_ambient_light([0.5, 0.5, 0.5])  # Brighter ambient light for better visibility
scene.add_directional_light([1, -1, -1], [1, 1, 1], shadow=False)
# Add a top-down light to better illuminate objects on the table
scene.add_directional_light([0, 0, -1], [0.7, 0.7, 0.7], shadow=False)
scene.add_ground(-0.4)

# ----------------------------------------------------------------------
# 2) Load robot and table
print("Loading robot and table...")
loader = scene.create_urdf_loader()
loader.fix_root_link = True  # Fix the robot's base

# Position the table
try:
    table_urdf_path = os.path.join(repo_root, "assets/objects/table.urdf")
    if not os.path.exists(table_urdf_path):
        print(f"⚠️ Warning: Table URDF not found at {table_urdf_path}")
        table = None
    else:
        table = loader.load(table_urdf_path)
        table_pos = [0.5, 0, 0]
        table.set_root_pose(sapien.Pose(table_pos, [1, 0, 0, 0]))
        print("✅ Table loaded successfully")
except Exception as e:
    print(f"Error loading table: {e}")
    table = None

# Position the robot
try:
    # Load Jaco2 robot
    robot_urdf_path = "/home/nb3227/RoboEXP/assets/robot/simple_arm/jaco2/jaco2.urdf"
    # Load the robot
    robot = loader.load(robot_urdf_path)
    # Position robot to better view the table
    robot_pos = [0.1, 0, 0.0]  # Closer to the table for better viewing
    robot.set_root_pose(sapien.Pose(robot_pos, [1, 0, 0, 0]))
    print("✅ Robot loaded successfully")

    # Print available joints and links for reference
    print("Available joints:")
    for i, joint in enumerate(robot.get_joints()):
        print(f"  - Joint {i}: {joint.get_name()} (DoF: {joint.get_dof()})")
    
    print("Available links:")
    for i, link in enumerate(robot.get_links()):
        print(f"  - Link {i}: {link.get_name()}")

    # Get current robot qpos and set initial positions
    current_qpos = robot.get_qpos()
    robot_dof = len(current_qpos)
    print(f"Robot has {robot_dof} degrees of freedom. Current qpos: {current_qpos}")
    
    # Try to set initial joint positions - adjust to look at the table
    try:
        init_qpos = [4.71, 2.84, 0, 1.2, 4.62, 4.48, 4.88, 0, 0, 0, 0, 0, 0]
        if len(init_qpos) == robot_dof:
            robot.set_qpos(init_qpos)
            print(f"Set initial joint positions for Jaco2: {init_qpos}")
        else:
            print(f"Warning: init_qpos length ({len(init_qpos)}) doesn't match robot DoF ({robot_dof})")
    except Exception as e:
        print(f"Error setting initial joint positions: {e}")
except Exception as e:
    print(f"Error loading robot: {e}")
    print("Cannot continue without robot. Exiting.")
    sys.exit(1)

# ----------------------------------------------------------------------
# 3) Load stapler from URDF with scaling
print("Loading stapler from URDF...")

# Create a separate loader just for the stapler with scaling
stapler_loader = scene.create_urdf_loader()
stapler_loader.fix_root_link = True  # Keep the same setting as the main loader
stapler_loader.scale = 0.1  # Scale to 10% of original size

# Load the stapler URDF with the scaled loader
stapler_urdf_path = os.path.join(repo_root, "assets/objects/stapler/mobility.urdf")
if not os.path.exists(stapler_urdf_path):
    print(f"⚠️ Error: Stapler URDF not found at {stapler_urdf_path}")
    print("Cannot continue without stapler. Exiting.")
    sys.exit(1)

# Load the stapler with the scaled loader
stapler = stapler_loader.load(stapler_urdf_path)

# Position the stapler on the table where it'll be visible
stapler_pos = [0.3, 0.0, 0.05]  # On the table, centered, closer to robot
stapler.set_root_pose(sapien.Pose(stapler_pos, [1, 0, 0, 0]))

# Try to "close" the stapler by setting qpos to zeros
try:
    stapler_qpos = stapler.get_qpos()
    if len(stapler_qpos) > 0:
        stapler.set_qpos(np.zeros_like(stapler_qpos))
        print(f"Set stapler to closed position: {stapler.get_qpos()}")
except Exception as e:
    print(f"Warning: Could not adjust stapler joints: {e}")

print("✅ Stapler loaded successfully from URDF with scale 0.1")
print(f"Stapler position: {stapler.get_pose().p}")

# ----------------------------------------------------------------------
# 4) Stabilize the robot using passive force compensation
print("Stabilizing robot position...")
for _ in range(50):  # Stabilization period
    try:
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
    except Exception as e:
        print(f"Warning: Error in passive force computation: {e}")
    scene.step()
print("Robot stabilized")

# ----------------------------------------------------------------------
# 5) Set up cameras
print("Setting up cameras...")

# Find the end effector link
end_link = robot.get_links()[-1]
print(f"Using {end_link.get_name()} as end effector for camera mounting")

# Create wrist camera
W_orig, H_orig = 512, 512  # Original camera resolution
wrist_offset = sapien.Pose([0, 0, 0.1], [1, 0, 0, 0])  # 10cm in front
wrist_cam = scene.add_mounted_camera(
    "wrist_cam", end_link, wrist_offset,
    width=W_orig, height=H_orig, fovy=np.deg2rad(60),
    near=0.1, far=100,
)

# Add global cameras for additional views
positions = [
    np.array([1.0, -1.0, 1.0]),  # Side view
    np.array([0.0, 0.0, 1.5]),   # Top view
]
target = np.array([0.3, 0.0, 0.03])  # Focus on the stapler position
dummy = scene.create_actor_builder().build_kinematic()
global_cams = []

for idx, eye in enumerate(positions):
    pose = sapien.Pose.from_transformation_matrix(get_pose_look_at(eye, target))
    cam = scene.add_mounted_camera(
        f"global_cam_{idx}", dummy, pose,
        width=W_orig, height=H_orig, fovy=np.deg2rad(60),
        near=0.1, far=100,
    )
    global_cams.append(cam)

# Set perception resolution
PERCEPTION_W, PERCEPTION_H = 128, 128

# ----------------------------------------------------------------------
# 6) Initialize Perception & Memory
print("Initializing perception and memory modules...")
output_dir = os.path.join(repo_root, "pipeline_output")

try:
    # Initialize perception with multiple object types, including stapler
    object_labels = "table . cabinet . box . chair . stapler . office_supply"
    robo_percept = RoboPercept(
        grounding_dict=object_labels, 
        lazy_loading=False,
        device="cuda", 
        use_sam_hq_in_segment=False
    )
    
    # Initialize memory system
    robo_memory = RoboMemory(
        lower_bound=[0, -0.8, -1], 
        higher_bound=[1, 0.5, 2],
        voxel_size=0.01, 
        real_camera=True, 
        base_dir=repo_root,
        similarity_thres=0.95, 
        iou_thres=0.01,
    )
    print("✅ Perception and memory modules initialized")
except Exception as e:
    print(f"Error initializing perception or memory: {e}")
    print("Cannot continue without perception and memory modules. Exiting.")
    sys.exit(1)

# ----------------------------------------------------------------------
# 7) End-effector poses to visit - adjusted to better view the stapler
poses = [
    [0.22, 0.0, 0.10, 1, 0, 0, 0],    # Centered, looking down at stapler
    [0.25, -0.05, 0.08, 1, 0, 0, 0],  # Left side view
    [0.25, 0.05, 0.08, 1, 0, 0, 0],   # Right side view
]

# ----------------------------------------------------------------------
# 8) Main processing loop
print("Starting main processing loop...")

for step, target_pose in enumerate(poses, 1):
    print(f"\n[{step}/{len(poses)}] Moving to pose {target_pose}")
    
    # Apply passive force before movement
    try:
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
    except Exception as e:
        print(f"Warning: Error in passive force computation: {e}")
    
    # Stabilize after movement
    for _ in range(20):
        try:
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
        except Exception as e:
            print(f"Warning: Error in passive force computation: {e}")
        scene.step()
    
    # Update rendering
    scene.update_render()
    
    # --- Capture and Perception/Memory Block ---
    print(f"   GPU Memory before capture: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # -- BEFORE you point the wrist cam, plan the arm pose -----------------
    ee_goal_pose = make_ee_pose_facing(stapler.get_pose())
    plan = planner.plan_to_pose(robot.get_qpos(), ee_goal_pose)
    if plan.success:
        for q in plan.joint_path:          # streamed execution
            robot.set_qpos(q)
            scene.step()
    else:
        print("⚠️ MPLib failed, falling back to old loop")
        # (leave your original 50‑step loop here as a fallback if you like)
    # ----------------------------------------------------------------------

    # Take pictures with all cameras
    point_wrist_cam_at(stapler.get_pose(), end_link, wrist_cam)

    wrist_cam.take_picture()
    for cam in global_cams:
        cam.take_picture()
    
    print("   📸 Camera pictures taken.")
    
    # Safely print camera and stapler positions
    print(f"   Wrist camera position: {wrist_cam.get_pose().p}")
    if stapler is not None:
        print(f"   Stapler position: {stapler.get_pose().p}")
        distance = np.linalg.norm(wrist_cam.get_pose().p - stapler.get_pose().p)
        print(f"   Distance between camera and stapler: {distance:.3f}m")
        # Calculate camera direction vector (assuming the camera looks along its local -z axis)
        camera_pose = wrist_cam.get_pose()
        # Use the rotation part of the pose to calculate direction
        # The pose matrix contains the full transformation
        pose_matrix = wrist_cam.get_model_matrix()  # Get the camera's model matrix
        # In a typical camera, the -z axis is the viewing direction
        # The 3rd column of the rotation part of the matrix represents the z axis direction
        camera_direction = -pose_matrix[:3, 2]  # Extract the 3rd column for z and negate for viewing direction
        # Normalize the direction vector
        camera_direction = camera_direction / np.linalg.norm(camera_direction)

        camera_to_stapler = stapler.get_pose().p - wrist_cam.get_pose().p
        camera_to_stapler = camera_to_stapler / np.linalg.norm(camera_to_stapler)
        # Calculate dot product to see if stapler is in camera view (positive means in front)
        dot_product = np.dot(camera_direction, camera_to_stapler)
        print(f"   Camera-stapler alignment (dot product): {dot_product:.3f} (higher is better)")
    else:
        print("   Stapler not found - cannot compute distance")
    
    # Get wrist camera textures
    rgb_texture_obj = None
    pos4_texture_obj = None
    
    try:
        rgb_texture_obj = wrist_cam.get_float_texture("Color")
        pos4_texture_obj = wrist_cam.get_float_texture("Position")
        print(f"   Successfully retrieved camera textures")
    except Exception as e:
        print(f"   Error getting camera textures: {e}")
        continue
        
    if rgb_texture_obj is None or pos4_texture_obj is None:
        print("   ❌ Camera textures are None. Skipping this step.")
        continue
    
    # Process the textures
    rgb_orig = rgb_texture_obj[..., :3]
    pos4_orig = pos4_texture_obj
    
    # Resize images for perception
    rgb_torch = torch.from_numpy(rgb_orig).permute(2, 0, 1).unsqueeze(0).cuda()
    pos4_torch = torch.from_numpy(pos4_orig).permute(2, 0, 1).unsqueeze(0).cuda()
    
    rgb_resized = F.interpolate(rgb_torch, size=(PERCEPTION_H, PERCEPTION_W), mode='bilinear', align_corners=False)
    pos4_resized = F.interpolate(pos4_torch, size=(PERCEPTION_H, PERCEPTION_W), mode='nearest')
    
    rgb_resized_np = rgb_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pos4_resized_np = pos4_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Create observation dictionary
    obs = dict(
        rgb = rgb_resized_np,
        position = pos4_resized_np[..., :3],
        mask = pos4_resized_np[..., 3] < 1,
        c2w = wrist_cam.get_model_matrix(),
        intrinsic = wrist_cam.get_intrinsic_matrix(),
        depths = pos4_resized_np[..., 2],
    )
    
    # Save camera view visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_orig)
    plt.title('Original Wrist Camera View')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_resized_np)
    plt.title('Resized View for Perception')
    plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, f'step_{step}_camera_view.png'))
    plt.close()
    
    # Also save the global camera views
    for i, cam in enumerate(global_cams):
        global_rgb = cam.get_float_texture("Color")[..., :3]
        plt.figure(figsize=(6, 6))
        plt.imshow(global_rgb)
        plt.title(f'Global Camera {i} View')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'step_{step}_global_cam_{i}.png'))
        plt.close()
    
    # Run perception
    print(f"   GPU Memory before perception: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    attrs = robo_percept.get_attributes_from_observations({"wrist": obs})
    print("   🔍 percept keys:", list(attrs.keys()))
    
    # Print detected objects for debugging
    if 'wrist' in attrs and 'pred_phrases' in attrs['wrist']:
        print("   🔍 Detected objects:", attrs['wrist']['pred_phrases'])
    
    print(f"   GPU Memory after perception: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Update memory with the correct labels including stapler
    object_level_labels = ["table", "cabinet", "box", "chair", "stapler", "office supply"]
    robo_memory.update_memory(
        observations = {"wrist": obs},
        observation_attributes = attrs,
        object_level_labels = object_level_labels,  # Match the grounding_dict
        filter_masks = {},
    )
    print("   💾 memory updated")
    print(f"   GPU Memory after memory update: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Try to visualize memory state
    try:
        if hasattr(robo_memory, 'visualize_current_state'):
            robo_memory.visualize_current_state(
                save_path=os.path.join(output_dir, f'step_{step}_memory_state.png')
            )
        print(f"   💾 Visualizations saved to {output_dir}")
    except Exception as vis_error:
        print(f"   Warning: Could not visualize memory state: {vis_error}")
        
    finally:
        # Memory cleanup
        locals_to_delete = [
            'obs', 'attrs', 'rgb_torch', 'pos4_torch', 'rgb_resized', 
            'pos4_resized', 'rgb_orig', 'pos4_orig', 
            'rgb_texture_obj', 'pos4_texture_obj'
        ]
        
        for var in locals_to_delete:
            if var in locals():
                try:
                    del locals()[var]
                except:
                    pass
                    
        torch.cuda.empty_cache()
        print("   ✨ GPU memory freed")
        print(f"   GPU Memory after clearing cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"   Max GPU Memory allocated so far: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# --- End of Main Loop ---
print("✅  Pipeline finished successfully.")
print(f"📁 Output visualizations have been saved to: {output_dir}")

# Final memory statistics
print("\nMemory Statistics:")
try:
    if hasattr(robo_memory, 'get_statistics'):
        stats = robo_memory.get_statistics()
        for key, value in stats.items():
            print(f"- {key}: {value}")
except Exception as e:
    print(f"Could not get memory statistics: {e}")

