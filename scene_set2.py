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

# Load stapler (scaled down)
stapler_loader = scene.create_urdf_loader()
stapler_loader.fix_root_link = True
stapler_loader.scale = 0.1
stapler = stapler_loader.load(os.path.join(repo_root, "assets/objects/camera/mobility.urdf"))
stapler.set_root_pose(sapien.Pose([0.3, 0.0, 0.05], [1, 0, 0, 0]))

# Load cabinet (larger scale)
cabinet_loader = scene.create_urdf_loader()
cabinet_loader.fix_root_link = True
cabinet_loader.scale = 0.3
cabinet = cabinet_loader.load(os.path.join(repo_root, "assets/objects/cabinet3/mobility.urdf"))
cabinet.set_root_pose(sapien.Pose([0.8, 0.0, 0.0], [1, 0, 0, 0]))  # Moved further back

# Stabilize scene
for _ in range(20):
    scene.step()

# Setup camera looking down at an angle
dummy = scene.create_actor_builder().build_kinematic()
camera_pos = [0.5, 0.0, 2]  # Centered above scene
dummy.set_pose(sapien.Pose(camera_pos))

# Different angle options (uncomment the one you want):

# Option 1: Straight down (90 degrees)
# rotation = [0.7071068, 0, 0.7071068, 0]

# Option 2: 45 degree angle
# rotation = [0.8536, 0, 0.3536, 0]

# Option 3: 30 degree angle
rotation = [0.5, 0, 0.5, 0]

# Create camera with selected angle
cam = scene.add_mounted_camera(
    "angled_cam", 
    dummy,
    sapien.Pose([0, 0, 0], rotation),  # Use the rotation you chose above
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

plt.imsave(os.path.join(scene_set_dir, "top_view2.png"), rgb)
print(f"✅ Scene photo saved to {scene_set_dir}/top_view2.png")
