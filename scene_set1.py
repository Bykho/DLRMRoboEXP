# scene_set1.py
import os
import matplotlib.pyplot as plt
import numpy as np
import sapien.core as sapien

def look_at(eye, target):
    """Create a look-at transformation matrix"""
    eye = np.asarray(eye)
    target = np.asarray(target)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Handle the case when looking straight down
    right = np.cross(forward, [0, 0, 1])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat

# Setup paths and scene
repo_root = os.path.dirname(os.path.abspath(__file__))
scene_set_dir = os.path.join(repo_root, "scene_set")
os.makedirs(scene_set_dir, exist_ok=True)

# Create scene
engine = sapien.Engine()
renderer = sapien.SapienRenderer(offscreen_only=True)
engine.set_renderer(renderer)

scene = engine.create_scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, -1, -1], [1, 1, 1])  # Light from front-top
scene.add_ground(-0.4)

# Load objects
loader = scene.create_urdf_loader()
loader.fix_root_link = True

# Load table
table = loader.load(os.path.join(repo_root, "assets/objects/table.urdf"))
table.set_root_pose(sapien.Pose([0.5, 0, 0], [1, 0, 0, 0]))

# Load robot
robot = loader.load("/home/nb3227/RoboEXP/assets/robot/simple_arm/jaco2/jaco2.urdf")
robot.set_root_pose(sapien.Pose([0.1, 0, 0], [1, 0, 0, 0]))
robot.set_qpos([4.71, 2.84, 0, 1.2, 4.62, 4.48, 4.88, 0, 0, 0, 0, 0, 0])

# Load cabinet (larger scale) and place it on the table
cabinet_loader = scene.create_urdf_loader()
cabinet_loader.fix_root_link = True
cabinet_loader.scale = 0.3
cabinet = cabinet_loader.load(os.path.join(repo_root, "assets/objects/cabinet_door/mobility.urdf"))
# Set cabinet on the table - table height is around 0.05, so we position cabinet at that height
cabinet.set_root_pose(sapien.Pose([0.8, 0.1, 0.18], [1, 0, 0, 0]))  # Positioned on top of table

# Get information about cabinet joints
num_joints = cabinet.dof
joint_names = [joint.name for joint in cabinet.get_joints()]
print(f"Cabinet has {num_joints} active joints")
print(f"All joint names: {joint_names}")

# Open all cabinet doors/drawers
if num_joints > 0:
    # Open all joints (assuming they're all doors/drawers)
    open_angles = [np.pi/2] * num_joints  # 90 degrees open for all joints
    cabinet.set_qpos(open_angles)
    print(f"Opened {num_joints} cabinet doors/drawers")
else:
    print("No movable doors/drawers found in cabinet")

# Load stapler (scaled down) and place it inside the open cabinet
stapler_loader = scene.create_urdf_loader()
stapler_loader.fix_root_link = True
stapler_loader.scale = 0.03
stapler = stapler_loader.load(os.path.join(repo_root, "assets/objects/stapler/mobility.urdf"))
# Position stapler inside the visible part of the open cabinet
stapler.set_root_pose(sapien.Pose([0.75, 0.0, 0.09], [1, 0, 0, 0]))

# Stabilize scene
for _ in range(20):
    scene.step()

# Setup camera with angled view towards cabinet front
dummy = scene.create_actor_builder().build_kinematic()
camera_pos = [0.5, -0.5, 0.3]  # Camera position

# Use a quaternion that we know works well (with a clear view of the cabinet front)
# This is a known working quaternion from previous attempts
rotation = [0.866, 0.3, 0.4, 0.2]

# Create camera with the selected angle
dummy.set_pose(sapien.Pose(camera_pos))
cam = scene.add_mounted_camera(
    "angled_cam", 
    dummy,
    sapien.Pose([0, 0, 0], rotation),
    width=512, 
    height=512,
    fovy=np.deg2rad(60),
    near=0.1,
    far=100
)

# Take and save photo
scene.update_render()
cam.take_picture()
rgb = cam.get_float_texture("Color")[..., :3]

plt.imsave(os.path.join(scene_set_dir, "top_view11.png"), rgb)
print(f"✅ Scene photo saved to {scene_set_dir}/top_view11.png")
