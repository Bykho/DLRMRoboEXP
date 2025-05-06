import sapien.core as sapien
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
from roboexp.utils import get_pose_look_at
import numpy as np
import open3d as o3d
import cv2
import time

# 1) Setup engine & scene
engine = sapien.Engine()
renderer = sapien.SapienRenderer(offscreen_only=True)
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([1, -1, -1], [1, 1, 1], shadow=False)
scene.add_ground(-0.4)

# 2) Load table and robot with fixed positions
loader = scene.create_urdf_loader()
loader.fix_root_link = True  # Fix the robot's base

# Instead of calculating dimensions, use hardcoded values for positioning
# Position the table
table = loader.load("/home/nb3227/RoboEXP/assets/objects/table.urdf")
table_pos = [0.5, 0, 0]
table.set_root_pose(sapien.Pose(table_pos, [1, 0, 0, 0]))

# Position the robot at the edge of the table
robot_pos = [0.2, 0, 0.0]  # Position robot near the table edge
robot = loader.load("/home/nb3227/RoboEXP/assets/robot/simple_arm/jaco2/jaco2.urdf")
robot.set_root_pose(sapien.Pose(robot_pos, [1, 0, 0, 0]))

# Print available joint names for reference
print("Available joints:")
for joint in robot.get_joints():
    print(f"  - {joint.get_name()}")

# Set initial joint positions - adjust these based on your robot's joint configuration
try:
    # Try to set the initial joint positions
    # These values will need to be adjusted for your specific robot model
    init_qpos = [0.0, 1.57, 0, 1.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if len(init_qpos) == len(robot.get_qpos()):
        robot.set_qpos(init_qpos)
    else:
        print(f"Warning: init_qpos length ({len(init_qpos)}) doesn't match robot DoF ({len(robot.get_qpos())})")
        print(f"Using default pose")
except Exception as e:
    print(f"Error setting initial joint positions: {e}")
    print("Using default robot pose")

# 3) Global cameras
positions = [
    np.array([1.0, -1.0, 1.0]),  # Side view
    np.array([0.0, 0.0, 1.5]),   # Top view
    np.array([1.0, 1.0, 1.0]),   # Opposite side view
]
target = np.array([0.5, 0.0, 0.3])  # Focus on the table center
fovy = np.deg2rad(60)
dummy = scene.create_actor_builder().build_kinematic()
global_cams = []
for idx, eye in enumerate(positions):
    pose = sapien.Pose.from_transformation_matrix(get_pose_look_at(eye, target))
    cam = scene.add_mounted_camera(f"global_cam_{idx}", dummy, pose,
                                  width=512, height=512, fovy=fovy,
                                  near=0.1, far=100)
    global_cams.append(cam)

# 4) Create a wrist-mounted camera
# Find the end effector link
print("Available links:")
for link in robot.get_links():
    print(f"  - {link.get_name()}")

# Try to find the end effector link
gripper_link = next((l for l in robot.get_links() if 'gripper' in l.get_name().lower()), None)
if gripper_link is None:
    # Fallback to the last link if no gripper link is found
    gripper_link = robot.get_links()[-1]
    print(f"Using {gripper_link.get_name()} as end effector (no gripper link found)")
else:
    print(f"Found gripper link: {gripper_link.get_name()}")

wrist_offset = sapien.Pose([0, 0, 0.1], [1, 0, 0, 0])

depth_cfg = StereoDepthSensorConfig()
depth_cfg.rgb_resolution = (512, 512)
depth_cfg.ir_resolution = (512, 512)
depth_cfg.rgb_intrinsic = np.array([[256.0, 0, 256.0], [0, 256.0, 256.0], [0, 0, 1]])
depth_cfg.ir_intrinsic = depth_cfg.rgb_intrinsic.copy()
depth_cfg.min_depth = 0.01
depth_cfg.max_depth = 5.0

wrist_depth = StereoDepthSensor("wrist_depth", scene, depth_cfg,
                               mount=gripper_link, pose=wrist_offset)

# Also add a simple wrist camera for compatibility with your other scripts
wrist_cam = scene.add_mounted_camera("wrist", gripper_link, wrist_offset,
                                    width=512, height=512, fovy=fovy,
                                    near=0.1, far=100)

# 5) Loop: step, render, and grab RGB+Depth → point cloud
print("Starting simulation loop...")

# Initial simulation stabilization period with passive force balancing
print("Stabilizing robot position...")
for _ in range(100):  # Run 100 steps to let the robot stabilize
    try:
        # Apply passive force compensation
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
    except Exception as e:
        print(f"Error in passive force computation: {e}")
    scene.step()
print("Robot stabilized")

# Main capture loop
for idx, cam in enumerate(global_cams):
    # Step simulation with passive force compensation
    for _ in range(4):
        try:
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
        except Exception as e:
            print(f"Error in passive force computation: {e}")
        scene.step()
    
    scene.update_render()

    # --- Global RGB view ---
    cam.take_picture()
    rgb = cam.get_float_texture("Color")[..., :3]
    frame = (rgb * 255).clip(0, 255).astype("uint8")
    cv2.imwrite(f"global_rgb_{idx}.png",
               cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Saved global_rgb_{idx}.png")

    # --- Wrist camera capture ---
    wrist_cam.take_picture()
    wrist_rgb = wrist_cam.get_float_texture("Color")[..., :3]
    wrist_frame = (wrist_rgb * 255).clip(0, 255).astype("uint8")
    cv2.imwrite(f"wrist_rgb_{idx}.png",
               cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR))
    print(f"Saved wrist_rgb_{idx}.png")

    # --- Wrist depth capture ---
    try:
        wrist_depth.take_picture()

        # first, grab color from the internal RGB camera
        color_w = wrist_depth._cam_rgb.get_float_texture("Color")[..., :3]

        # **compute the stereo depth before calling get_depth()**
        wrist_depth.compute_depth()
        depth_map = wrist_depth.get_depth()  # H×W float depth

        # now grab the 3D positions from the underlying camera
        positions3d = wrist_depth._cam_rgb.get_float_texture("Position")[..., :3]
        positions3d[..., 2] = -depth_map  # flip sign to match conventions

        # Build Open3D point cloud (only valid pixels)
        mask = (depth_map > depth_cfg.min_depth) & (depth_map < depth_cfg.max_depth)
        xyz = positions3d[mask]
        rgb_flat = color_w[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb_flat)

        # Voxel-downsample so it's "mini"
        pcd_mini = pcd.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud(f"wrist_pcd_{idx}.ply", pcd_mini)
        print(f"Saved wrist_pcd_{idx}.ply ({len(pcd_mini.points)} points)")
    except Exception as e:
        print(f"Error in depth processing: {e}")

print("All views + mini point-clouds saved.")