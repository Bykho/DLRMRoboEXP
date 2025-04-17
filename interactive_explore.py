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
import sapien
import numpy as np
import cv2

# Set environment variables for NVIDIA Vulkan rendering
os.environ["SAPIEN_HEADLESS"] = "1"
os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

# Make sure we're using the virtual display
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":99"

def save_frame(frame, frame_dir, frame_num):
    """Save a frame as an image."""
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    frame_path = os.path.join(frame_dir, f"frame_{frame_num:06d}.png")
    # Convert from RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(frame_path, frame_bgr)

def explore(robo_decision, robo_act, frame_dir):
    """Test exploration with saved observations"""
    frame_num = 0
    
    # Define a sequence of safe poses to test
    test_poses = [
        # Home position
        np.array([0.2, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0]),
        # Slightly to the left
        np.array([0.2, 0.1, 0.15, 1.0, 0.0, 0.0, 0.0]),
        # Slightly to the right
        np.array([0.2, -0.1, 0.15, 1.0, 0.0, 0.0, 0.0]),
    ]
    
    for pose in test_poses:
        try:
            print(f"\nTesting pose: {pose}")
            success = robo_act.robo_exp.run_action(
                action_code=1,
                action_parameters=pose,
                iteration=1000
            )
            
            if not success:
                print("Movement failed, skipping observation")
                continue
                
            # Try to get observations
            try:
                obs = robo_act.get_observations_update_memory(
                    update_scene_graph=True,
                    visualize=True
                )
                
                if obs is not None and 'rgb' in obs:
                    print("Successfully captured observation")
                    # Save the frame if directory is provided
                    if frame_dir:
                        save_frame(obs['rgb'], frame_dir, frame_num)
                        frame_num += 1
                        
                    # Process the observation with robo_decision
                    action = robo_decision.get_action(obs)
                    print(f"Decided action: {action}")
                    
                    # Execute the decided action
                    if action["action"] == "open_close":
                        robo_act.open_close_gripper()
                    elif action["action"] == "pick":
                        robo_act.pick_object(action["target_pos"])
                else:
                    print("Failed to get valid observation data")
                    
            except Exception as e:
                print(f"Error during observation capture: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Error during pose testing: {str(e)}")
            continue
            
    print("\nExploration sequence completed")

def run(base_dir, REPLAY_FLAG=False):
    # Load simulation configuration
    with open("config/simulation_config.json", "r") as f:
        sim_config = json.load(f)
    
    # Initialize the robot in simulation
    # Remove conflicting environment variables
    if 'LIBGL_ALWAYS_INDIRECT' in os.environ:
        del os.environ['LIBGL_ALWAYS_INDIRECT']
    
    robo_exp = RobotExploration(
        data_path=sim_config["data_path"],
        robot_conf=sim_config["robot_conf"],
        objects_conf=sim_config["objects_conf"],
        ray_tracing=True,
        balance_passive_force=True,
        offscreen_only=True,
        gt_depth=False,
        has_gripper=False,  # Disable gripper constraints since we have a simple gripper
        control_mode="mplib",
    )

    # Initialize the memory module
    robo_memory = RoboMemory(
        lower_bound=[0, -0.8, -1],
        higher_bound=[1, 0.5, 2],
        voxel_size=0.01,
        real_camera=True,
        base_dir=base_dir,
        similarity_thres=0.95,
        iou_thres=0.01,
    )

    # Set the labels
    object_level_labels = [
        "table",
        "refrigerator",
        "cabinet",
        "can",
        "doll",
        "plate",
        "spoon",
        "fork",
        "hamburger",
        "condiment",
    ]
    part_level_labels = ["handle"]

    grounding_dict = (
        " . ".join(object_level_labels) + " . " + " . ".join(part_level_labels)
    )
    # Initialize the perception module
    robo_percept = RoboPercept(grounding_dict=grounding_dict, lazy_loading=False)
    # Initialize the action module
    robo_act = RoboAct(
        robo_exp,
        robo_percept,
        robo_memory,
        object_level_labels,
        base_dir=base_dir,
        REPLAY_FLAG=REPLAY_FLAG,
    )
    # Initialize the decision module
    robo_decision = RoboDecision(robo_memory, base_dir, REPLAY_FLAG=REPLAY_FLAG)

    # Create frames directory
    frame_dir = os.path.join(base_dir, "frames")
    explore(robo_decision, robo_act, frame_dir)


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{current_time}"
    REPLAY_FLAG = False
    if not os.path.exists(base_dir):
        # Create directory if it doesn't exist
        os.makedirs(base_dir)
    run(base_dir, REPLAY_FLAG=REPLAY_FLAG)
