import os
import sys
import json
import cv2
import numpy as np
import math
import sapien.core as sapien

from roboexp import RobotExploration
from roboexp.utils import get_pose_look_at

# 1) Repo root on PYTHONPATH
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

# 2) Load sim config & pick only the table
config_path = os.path.join(parent_dir, "config", "simulation_config.json")
with open(config_path, "r") as f:
    sim_cfg = json.load(f)
objs      = [o for o in sim_cfg["objects_conf"] if o.get("name") == "table"]
data_path = os.path.join(parent_dir, sim_cfg["data_path"])

# 3) Init RobotExploration in RT mode (no IBL)
robo = RobotExploration(
    data_path=data_path,
    robot_conf=sim_cfg["robot_conf"],
    objects_conf=objs,
    ray_tracing=True,
    balance_passive_force=True,
    offscreen_only=True,
    gt_depth=False,
    has_gripper=False,
    control_mode="mplib",
)

# 4) Simple ambient only, RT shading  
robo.scene.set_ambient_light([0.3, 0.3, 0.3])
sapien.render_config.camera_shader_dir    = "rt"
sapien.render_config.viewer_shader_dir    = "rt"
sapien.render_config.rt_samples_per_pixel = 4
sapien.render_config.rt_use_denoiser      = False

# 5) Mount the wrist camera  
end_link     = robo.robot.get_links()[-1]
wrist_offset = sapien.Pose([0, 0, 0.1], [1, 0, 0, 0])
W, H         = 512, 512
fovy         = np.deg2rad(60)

wrist_cam = robo.scene.add_mounted_camera(
    "wrist_cam", end_link, wrist_offset,
    width=W, height=H, fovy=fovy, near=0.1, far=100,
)

# 6) Mount the third-person camera (high behind view)  
thirdpov_dummy  = robo.scene.create_actor_builder().build_kinematic()
thirdpov_eye    = np.array([0.0, -0.8, 1.2])
thirdpov_target = np.array([0.2,  0.0, 0.2])
tp_pose = sapien.Pose.from_transformation_matrix(
    get_pose_look_at(thirdpov_eye, thirdpov_target)
)
thirdpov_cam = robo.scene.add_mounted_camera(
    "thirdpov_cam", thirdpov_dummy, tp_pose,
    width=W, height=H, fovy=fovy, near=0.1, far=100,
)

# 7) Video writers  
fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
out_wrist    = cv2.VideoWriter("robo_wrist_sweep.mp4", fourcc, 10.0, (W, H))
out_thirdppo = cv2.VideoWriter("robo_thirdpov_sweep.mp4", fourcc, 10.0, (W, H))

# 8) Setup joint sweep  
joints = robo.robot.get_active_joints()  # [base_to_arm, arm_to_gripper]
qlimits = robo.robot.get_qlimits()       # [[min0,max0],[min1,max1]]

joint_idx   = 0                         # sweep the base_to_arm joint
qmin, qmax  = qlimits[joint_idx]
other_idx   = 1 - joint_idx
# freeze other joint at its midpoint
fixed_mid   = 0.5 * (qlimits[other_idx][0] + qlimits[other_idx][1])

n_steps = 120  # ~12 seconds at 10 FPS
angles = np.linspace(qmin, qmax, n_steps)

# 9) Sweep and record
for i, angle in enumerate(angles):
    # build desired qpos
    qpos = np.array([fixed_mid, fixed_mid], dtype=float)
    qpos[joint_idx] = angle

    # drive the joints (smooth PID)
    for idx, joint in enumerate(joints):
        joint.set_drive_target(qpos[idx])

    # step and render
    robo.scene.step()
    robo.scene.update_render()

    # wrist view
    wrist_cam.take_picture()
    wrgb = wrist_cam.get_float_texture("Color")[..., :3]
    wframe = (wrgb * 255).clip(0,255).astype("uint8")
    out_wrist.write(cv2.cvtColor(wframe, cv2.COLOR_RGB2BGR))

    # third-person view
    thirdpov_cam.take_picture()
    trg = thirdpov_cam.get_float_texture("Color")[..., :3]
    tframe = (trg * 255).clip(0,255).astype("uint8")
    out_thirdppo.write(cv2.cvtColor(tframe, cv2.COLOR_RGB2BGR))

    print(f"Frame {i+1}/{n_steps}: joint[{joint_idx}] = {angle:.3f}")

# 10) Cleanup
out_wrist.release()
out_thirdppo.release()
print("🎬 Done! Swept joint and saved videos:")
print("    - robo_wrist_sweep.mp4")
print("    - robo_thirdpov_sweep.mp4")
