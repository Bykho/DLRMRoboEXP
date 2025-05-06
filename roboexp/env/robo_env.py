import sapien.core as sapien
from sapien.utils.viewer import Viewer
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
import mplib
import numpy as np
import open3d as o3d
import time
from roboexp.utils import get_pose_look_at


def NONE_FUNC():
    return True


class RobotExploration:
    def __init__(
        self,
        data_path,
        robot_conf,
        objects_conf,
        ray_tracing=False,
        balance_passive_force=True,
        offscreen_only=True,
        gt_depth=False,
        has_gripper=True,
        control_mode="mplib",
        timestep=1 / 250.0,
        ground_altitude=-0.4,
    ):
        self.data_path = data_path
        self.robot_conf = robot_conf
        self.objects_conf = objects_conf
        self.ray_tracing = ray_tracing
        self.balance_passive_force = balance_passive_force
        self.offscreen_only = offscreen_only
        self.gt_depth = gt_depth
        self.has_gripper = has_gripper
        self.control_mode = control_mode
        self.timestep = timestep
        self.ground_altitude = ground_altitude
        # Below two will automatially change based on the use case
        self.use_point_cloud = False
        self.use_attach = False
        # Some parameters for the robot control
        self.ROBOT_STOP_FLAG = False
        self.robot_move_unit = 0.02
        # Initialize the observation
        self.current_observations = None
        # Initialize the gripper state
        self.gripper_state = 1
        # Record the target pose
        self.target_pose = None
        # Close the interactive mode for now
        self.stop_loop = False

        # Initialize the engine
        self.engine = sapien.Engine()
        self.engine.set_log_level("error")
        self.renderer = sapien.SapienRenderer(offscreen_only=self.offscreen_only)
        self.engine.set_renderer(self.renderer)
        # Creat the scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(self.timestep)

        # Initialize the environment
        self.init_env()
        # Initialize the robot control
        self.init_robot_control()

        if not self.offscreen_only:
            # Set up the GUI
            self.init_viewer()

    def init_viewer(self):
        # Set up the GUI
        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(-0.5, 0, 1.5)
        self.viewer.set_camera_rpy(r=0, p=-0.7853981, y=0)
        self.viewer.toggle_camera_lines(False)
        self.viewer.toggle_axes(False)

    def init_env(self):
        # Minimal lighting setup - just ambient and one directional light
        self.scene.set_ambient_light([0.6, 0.6, 0.6])  # Increased ambient light
        #self.scene.add_directional_light([0, 1, -1], [0.8, 0.8, 0.8])  # Brighter directional light
        
        # Add three mounted cameras
        self.camera_positions = {
            "camera_0": np.array([0.3, 0, 0.9]),
            "camera_1": np.array([0.3, 0.6, 0.4]),
            "camera_2": np.array([0.3, -0.6, 0.4]),
            "camera_3": np.array([0.3, 0.25, 0.9]),
            "camera_4": np.array([0.3, -0.25, 0.9]),
        }
        target = np.array([0.8, 0.0, 0.3])
        self.cameras = {}
        if self.ray_tracing and self.gt_depth:
            self.cameras_rt = {}
        for name, camera_position in self.camera_positions.items():
            self.cameras[name] = self._add_mount_camera(
                name=name,
                pose=sapien.Pose.from_transformation_matrix(
                    get_pose_look_at(eye=np.array(camera_position), target=target)
                ),
                ray_tracing=False,
            )
            if self.ray_tracing and self.gt_depth:
                self.cameras_rt[name] = self._add_mount_camera(
                    name=f"{name}_rt",
                    pose=sapien.Pose.from_transformation_matrix(
                        get_pose_look_at(eye=np.array(camera_position), target=target)
                    ),
                    ray_tracing=True,
                )

        # Add ground
        ground_material = self.renderer.create_material()
        ground_material.base_color = np.array([256, 256, 256, 256]) / 256
        ground_material.specular = 0.5
        self.scene.add_ground(self.ground_altitude, render_material=ground_material)
        # Load the objects
        self.articulated_models = []
        self.models = []
        for object in self.objects_conf:
            if "urdf_path" in object:
                # Load the articulated object from SAPIEN dataset
                self.articulated_models.append(
                    self._load_sapien_model(
                        path=f"{self.data_path}/{object['urdf_path']}",
                        pose=sapien.Pose(object["init_pos"], object["init_rot"]),
                        fix_root_link=object["fix_root_link"]
                        if "fix_root_link" in object
                        else True,
                        scale=object["scale"] if "scale" in object else 1,
                        name=object["name"] if "name" in object else "Undefined",
                    )
                )
            elif "model_path" in object:
                # Load other objects from the object_path
                self.models.append(
                    self._load_model(
                        path=f"{self.data_path}/{object['model_path']}",
                        scale=object["scale"] if "scale" in object else 1,
                        dynamic=object["dynamic"] if "dynamic" in object else True,
                        pose=sapien.Pose(object["init_pos"], object["init_rot"]),
                        name=object["name"] if "name" in object else "Undefined",
                    )
                )
            else:
                print(f"Not supported format for {object}")
                raise NotImplementedError
        # Load the robot
        self.robot = self._load_robot(
            path=f"{self.data_path}/{self.robot_conf['urdf_path']}",
            pose=sapien.Pose(self.robot_conf["init_pos"], self.robot_conf["init_rot"]),
        )

        # Add the wrist camera
        wrist_transform = np.array(
            [[0, 0, 1, 0.1], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
        )
        wrist_pose = sapien.Pose.from_transformation_matrix(wrist_transform)
        # Get the gripper link for camera mounting
        gripper_link = next(link for link in self.robot.get_links() if link.name == "gripper_link")
        self.cameras["wrist"] = self._add_mount_camera(
            name="wrist",
            pose=wrist_pose,
            camera_mount_actor=gripper_link,
            ray_tracing=False,
        )
        if self.ray_tracing and self.gt_depth:
            self.cameras_rt["wrist"] = self._add_mount_camera(
                name="wrist_rt",
                pose=wrist_pose,
                camera_mount_actor=gripper_link,
                ray_tracing=True,
            )

    def _add_mount_camera(
        self, name, pose=sapien.Pose(), camera_mount_actor=None, ray_tracing=True
    ):
        if ray_tracing:
            sapien.render_config.camera_shader_dir = "rt"
            sapien.render_config.viewer_shader_dir = "rt"
            sapien.render_config.rt_samples_per_pixel = 16
            sapien.render_config.rt_use_denoiser = True
        else:
            sapien.render_config.camera_shader_dir = "ibl"
        self.scene.update_render()
        if camera_mount_actor is None:
            camera_mount_actor = self.scene.create_actor_builder().build_kinematic()
            camera_mount_actor.set_pose(pose)
            local_pose = sapien.Pose()
        else:
            local_pose = pose
        if self.gt_depth:
            # The depth image is with no noise
            near, far = 0.01, 2
            width, height = 512, 512
            fov = np.pi / 2
            camera = self.scene.add_mounted_camera(
                name=name,
                actor=camera_mount_actor,
                pose=local_pose,  # relative to the mounted actor
                width=width,
                height=height,
                fovy=fov,
                near=near,
                far=far,
            )
        else:
            # The depth image is with noise
            camera_config = StereoDepthSensorConfig()
            # Set the resolution and intrinsic parameters
            camera_config.rgb_resolution = (512, 512)
            camera_config.rgb_intrinsic = np.array(
                [[256.0, 0.0, 256.0], [0.0, 256.0, 256.0], [0.0, 0.0, 1.0]]
            )
            camera_config.ir_resolution = (512, 512)
            camera_config.ir_intrinsic = np.array(
                [[256.0, 0.0, 256.0], [0.0, 256.0, 256.0], [0.0, 0.0, 1.0]]
            )
            camera_config.min_depth = 0.01
            camera_config.max_depth = 5
            camera = StereoDepthSensor(
                name,
                self.scene,
                camera_config,
                mount=camera_mount_actor,
                pose=local_pose,
            )  # pose is relative to mount
        return camera

    def _load_model(
        self, path, pose=sapien.Pose(), dynamic=True, scale=1, name="Undefined"
    ):
        material = self.scene.create_physical_material(
            **dict(static_friction=1, dynamic_friction=1, restitution=0)
        )
        builder = self.scene.create_actor_builder()

        if type(scale) == list:
            scale = np.array(scale)
        else:
            scale = np.ones(3) * scale

        builder.add_collision_from_file(filename=path, scale=scale, material=material)
        builder.add_visual_from_file(filename=path, scale=scale)
        if dynamic:
            model = builder.build(name=name)
        else:
            model = builder.build_static(name=name)
        model.set_pose(pose)
        return model

    def _load_sapien_model(
        self,
        path,
        pose=sapien.Pose(),
        dynamic=True,
        fix_root_link=True,
        scale=1,
        name="Undefined",
    ):
        # Refer to the code in https://github.com/haosulab/ManiSkill2/blob/main/mani_skill2/envs/ms1/open_cabinet_door_drawer.py#L50
        urdf_config = {
            "material": self.scene.create_physical_material(
                **dict(static_friction=1, dynamic_friction=1, restitution=0)
            )
        }

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = fix_root_link
        loader.scale = scale
        loader.load_multiple_collisions_from_file = True
        if dynamic:
            model = loader.load(path, config=urdf_config)
        else:
            model = loader.load_kinematic(path, config=urdf_config)
        model.set_root_pose(pose)

        # Ignore collision within the articulation to avoid impact from imperfect collision shapes.
        # The legacy version only ignores collision of child links of active joints.
        for link in model.get_links():
            for s in link.get_collision_shapes():
                g0, g1, g2, g3 = s.get_collision_groups()
                s.set_collision_groups(g0, g1, g2 | 1 << 31, g3)

        # Set the physical parameters
        for joint in model.get_active_joints():
            joint.set_friction(np.random.uniform(0.05, 0.15))
            joint.set_drive_property(stiffness=0, damping=np.random.uniform(5, 20))

        # Set joint positions to lower bounds
        qlimits = model.get_qlimits()  # [N, 2]
        assert not np.isinf(qlimits).any(), qlimits
        qpos = np.ascontiguousarray(qlimits[:, 0])
        model.set_qpos(qpos)

        model.set_name(name)
        assert model, f"Partnetsim model {name} not loaded."
        return model

    def _load_robot(self, path, pose=sapien.Pose(), fix_root_link=True):
        # Set the friction for the robot
        urdf_config = {
            "material": self.scene.create_physical_material(
                **dict(static_friction=1, dynamic_friction=1, restitution=0)
            ),
            "link": {
                "right_finger": {
                    "material": self.scene.create_physical_material(
                        **dict(
                            static_friction=1000, dynamic_friction=1000, restitution=0
                        )
                    )
                },
                "left_finger": {
                    "material": self.scene.create_physical_material(
                        **dict(
                            static_friction=1000, dynamic_friction=1000, restitution=0
                        )
                    )
                },
            },
        }
        loader = self.scene.create_urdf_loader()
        builder = loader.load_file_as_articulation_builder(path, config=urdf_config)
        # Disable self collision for simplification
        for link_builder in builder.get_link_builders():
            link_builder.set_collision_groups(1, 1, 2, 0)
        robot = builder.build(fix_root_link=True)
        robot.set_root_pose(pose)
        robot.set_name("Robot")
        if self.has_gripper:
            # Set the gripper constraint
            self._add_gripper_constraint(robot)
        assert robot, "Robot not loaded."
        return robot

    def _add_gripper_constraint(self, robot):
        # Follow the code in https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py#L111
        # Change the joint name based on the URDF
        outer_knuckle = next(
            j
            for j in robot.get_active_joints()
            if j.name == "right_outer_knuckle_joint"
        )
        outer_finger = next(
            j for j in robot.get_active_joints() if j.name == "right_finger_joint"
        )
        inner_knuckle = next(
            j
            for j in robot.get_active_joints()
            if j.name == "right_inner_knuckle_joint"
        )

        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        T_pw = pad.pose.inv().to_transformation_matrix()
        p_w = (
            outer_finger.get_global_pose().p
            + inner_knuckle.get_global_pose().p
            - outer_knuckle.get_global_pose().p
        )
        T_fw = lif.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f), pad, sapien.Pose(p_p)
        )
        right_drive.lock_motion(1, 1, 1, 0, 0, 0)

        outer_knuckle = next(
            j for j in robot.get_active_joints() if j.name == "drive_joint"
        )
        outer_finger = next(
            j for j in robot.get_active_joints() if j.name == "left_finger_joint"
        )
        inner_knuckle = next(
            j for j in robot.get_active_joints() if j.name == "left_inner_knuckle_joint"
        )

        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        T_pw = pad.pose.inv().to_transformation_matrix()
        p_w = (
            outer_finger.get_global_pose().p
            + inner_knuckle.get_global_pose().p
            - outer_knuckle.get_global_pose().p
        )
        T_fw = lif.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f), pad, sapien.Pose(p_p)
        )
        left_drive.lock_motion(1, 1, 1, 0, 0, 0)

    def _balance_passive_force(self):
        qf = self.robot.compute_passive_force(
            gravity=True, coriolis_and_centrifugal=True, external=False
        )
        self.robot.set_qf(qf)

    def _init_action(self):
        self.actions_position_list = []

    def _do_action(self):
        # Update the robot based on the current_steps
        if len(self.actions_position_list) != 0:
            action_position = self.actions_position_list.pop(0)
            for j in range(len(self.robot_active_joints)):
                self.robot_active_joints[j].set_drive_target(action_position[j])

            if len(self.actions_position_list) == 0:
                print("Robot no action")

    # Run the interactive simulation
    def run_interactive(self):
        if not self.offscreen_only and self.viewer.closed:
            # Reinitialize the viewer if the viewer is closed
            self.init_viewer()
        print("Running interactive Mode")

        def _key_detect():
            # Support interactive control with viewer
            if len(self.actions_position_list) == 0:
                action = None
                movement = None
                action = None
                if self.viewer.window.key_down("right"):
                    print(f"Moving the end effector: y + {self.robot_move_unit}")
                    movement = np.array([0, -self.robot_move_unit, 0])
                elif self.viewer.window.key_down("left"):
                    print(f"Moving the end effector: y - {self.robot_move_unit}")
                    movement = np.array([0, self.robot_move_unit, 0])
                elif self.viewer.window.key_down("up"):
                    print(f"Moving the end effector: x + {self.robot_move_unit}")
                    movement = np.array([self.robot_move_unit, 0, 0])
                elif self.viewer.window.key_down("down"):
                    print(f"Moving the end effector: x - {self.robot_move_unit}")
                    movement = np.array([-self.robot_move_unit, 0, 0])
                elif self.viewer.window.key_down("j"):
                    print(f"Moving the end effector: z + {self.robot_move_unit}")
                    movement = np.array([0, 0, self.robot_move_unit])
                elif self.viewer.window.key_down("k"):
                    print(f"Moving the end effector: z - {self.robot_move_unit}")
                    movement = np.array([0, 0, -self.robot_move_unit])
                elif self.viewer.window.key_down("i"):
                    print("Take Pics and get the PC")
                    self.get_observations(save_image=True)
                    self.visualize_observations()
                elif self.viewer.window.key_down("u"):
                    print("Take Pics and get the PC")
                    self.get_observations(wrist_only=True, save_image=True, gt_seg=True)
                    self.visualize_observations()
                elif self.viewer.window.key_down("o"):
                    if self.gripper_state == 0:
                        print("Open the gripper")
                        self.gripper_state = 1
                        self._run(iteration=100)
                    elif self.gripper_state == 1:
                        print("Close the gripper")
                        self.gripper_state = 0
                        self._run(iteration=100)
                elif self.viewer.window.key_down("m"):
                    # print("Display the current observation")
                    self.visualize_observations()
                elif self.viewer.window.key_down("p"):
                    print("Reset the robot qpos")
                    self.robot_reset()
                    self._run(iteration=100)
                if movement is not None:
                    # Update the target pose
                    new_position = self.target_pose[:3] + movement
                    new_rotation = self.target_pose[3:]
                    self.target_pose[:3] = new_position
                    self.target_pose[3:] = new_rotation
                    action = self.robot_move_to_pose(
                        np.concatenate([new_position, new_rotation])
                    )
                    print(
                        f"DETAIL: Current end effector pose: {self.get_end_effector_pose()}"
                    )
                    print(
                        f"DETAIL: Target end effector pose: {new_position} {new_rotation}"
                    )
                    time.sleep(0.5)
                if action is not None:
                    if action == -1:
                        print("Invalid action")
                        return False
                    print("Robot Acting")
                    self.actions_position_list = list(action["position"])

            return True

        self._run(
            before_loop_fn=self._init_action,
            before_render_fn=self._do_action,
            after_render_fn=_key_detect,
            iteration=-1,
        )

    # Only run one action each time
    def run_action(self, action_code, action_parameters, iteration=1000):
        """Run robot action with validation"""
        print(f"Running action {action_code} with parameters {action_parameters}")
        
        # Validate target pose for movement actions
        if action_code == 1:  # Movement action
            is_valid, message = self._validate_target_pose(action_parameters)
            if not is_valid:
                print(f"Invalid target pose: {message}")
                return False
                
        # Convert parameters to numpy array if needed
        if isinstance(action_parameters, list):
            action_parameters = np.array(action_parameters)
            
        if action_code == 1:
            # Movement action
            try:
                # Get current end effector pose for debugging
                gripper_link = next(link for link in self.robot.get_links() if link.name == "gripper_link")
                current_pose = gripper_link.get_pose()
                print(f"Current end effector: pos={current_pose.p}, quat={current_pose.q}")
                
                # Try to find IK solution
                target_pos = action_parameters[:3]
                target_quat = action_parameters[3:]
                print(f"Target: pos={target_pos}, quat={target_quat}")
                
                # Get the gripper link index for IK
                gripper_link_idx = next(i for i, link in enumerate(self.robot.get_links()) if link.name == "gripper_link")
                
                inverse_kinematics = self.IK_solver.compute_inverse_kinematics(
                    link_index=gripper_link_idx,
                    pose=sapien.Pose(target_pos, target_quat),
                    initial_qpos=self.robot.get_qpos(),
                )[0]
                
                print(f"Found IK solution: {inverse_kinematics}")
                optimized_qpos = self._optimize_IK(inverse_kinematics)
                print(f"Optimized joint angles: {optimized_qpos}")
                
                # Set the joint positions
                for i, joint in enumerate(self.robot_active_joints):
                    joint.set_drive_target(optimized_qpos[i])
                    
                # Wait for movement to complete
                for _ in range(iteration):
                    self.scene.step()
                    
                # Verify final position
                final_pose = gripper_link.get_pose()
                pos_error = np.linalg.norm(final_pose.p - target_pos)
                print(f"Final position error: {pos_error:.6f}m")
                
                return True
                
            except Exception as e:
                print(f"Movement action failed: {str(e)}")
                return False
                
        elif action_code == 2:
            # Handle other action codes here
            pass
            
        return False

    # The function to run the simulation
    def _run(
        self,
        before_loop_fn=NONE_FUNC,
        before_render_fn=NONE_FUNC,
        after_render_fn=NONE_FUNC,
        iteration=100,
    ):
        step = 0
        before_loop_fn()
        while True:
            if iteration != -1 and step >= iteration:
                break
            self._balance_passive_force()
            self.scene.step()
            if self.gripper_state == 0:
                self.robot_close_gripper()
            else:
                self.robot_open_gripper()
            before_render_fn()
            if step % 4 == 0:
                self.scene.update_render()
                if not self.offscreen_only:
                    self.viewer.render()
                    if self.viewer.closed:
                        break
                if not after_render_fn():
                    break
            step += 1

    # Funciton to take picture, return the observations
    def get_observations(
        self, wrist_only=False, save_image=False, gt_seg=False, **kwargs
    ):
        try:
            # Here we support two modes for now: wrist only or all other cameras
            if wrist_only:
                allow_camera_names = ["wrist"]
            else:
                allow_camera_names = list(self.camera_positions.keys())
            observations = {}
            
            # Update render configuration for stability
            if self.ray_tracing:
                sapien.render_config.camera_shader_dir = "rt"
                sapien.render_config.viewer_shader_dir = "rt"
                sapien.render_config.rt_samples_per_pixel = 4  # Reduced for stability
                sapien.render_config.rt_use_denoiser = False  # Disabled for stability
            else:
                sapien.render_config.camera_shader_dir = "ibl"
            
            # Force scene update before capturing
            self.scene.update_render()
            
            for name, camera in self.cameras.items():
                if name not in allow_camera_names:
                    continue
                    
                try:
                    camera.take_picture()
                    
                    # Get the RGB and position data
                    if self.gt_depth:
                        color = camera.get_float_texture("Color") if not self.ray_tracing else self.cameras_rt[name].get_float_texture("Color")
                        position = camera.get_float_texture("Position")
                    else:
                        color = camera._cam_rgb.get_float_texture("Color")
                        camera.compute_depth()
                        position = camera._cam_rgb.get_float_texture("Position")
                        depth = camera.get_depth()
                        position[..., 2] = -depth
                    
                    # Get camera matrices
                    model_matrix = camera.get_model_matrix()
                    intrinsic_matrix = camera.get_intrinsic_matrix()
                    
                    # Save image if requested
                    if save_image:
                        from PIL import Image
                        color_img = (color * 255).clip(0, 255).astype("uint8")
                        color_pil = Image.fromarray(color_img)
                        color_pil.save(f"{name}_color.png")
                    
                    # Store observation data
                    observations[name] = {
                        "rgb": color[..., :3],
                        "position": position[..., :3],
                        "mask": position[..., 3] < 1,
                        "c2w": model_matrix,
                        "intrinsic": intrinsic_matrix,
                    }
                except Exception as e:
                    print(f"Warning: Failed to capture from camera {name}: {str(e)}")
                    continue
                    
            if not observations:
                raise RuntimeError("Failed to capture any observations")
                
            self.current_observations = observations
            return observations
            
        except Exception as e:
            print(f"Error capturing observations: {str(e)}")
            return None

    # This function is purely for quick visualization
    def visualize_observations(self):
        if self.current_observations is None:
            print("No current observation")
            return
        total_pcd = o3d.geometry.PointCloud()
        for name, observation in self.current_observations.items():
            color = observation["rgb"]
            position = observation["position"]
            mask = observation["mask"]
            c2w = observation["c2w"]

            # OpenGL/Blender: y up and -z forward
            pc_position = position[mask]
            pc_color = color[mask]
            points_world = pc_position @ c2w[:3, :3].T + c2w[:3, 3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            pcd.colors = o3d.utility.Vector3dVector(pc_color)

            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            o3d.visualization.draw_geometries([pcd])
            total_pcd += pcd

        total_pcd = total_pcd.voxel_down_sample(voxel_size=0.01)
        o3d.visualization.draw_geometries([total_pcd])

    def get_end_effector_pose(self):
        # Get the gripper link pose as the end effector pose
        gripper_link = next(link for link in self.robot.get_links() if link.name == "gripper_link")
        pose = gripper_link.get_pose()
        print(f"DEBUG: Current end effector pose: position={pose.p}, quaternion={pose.q}")
        return pose

    def get_gripper_position(self):
        # For our simple robot, the gripper position is the same as the end effector position
        return self.get_end_effector_pose().p

    def init_robot_control(self):
        print("\nDEBUG: Initializing robot control")
        print(f"DEBUG: Robot configuration: {self.robot_conf}")
        if "init_qpos" in self.robot_conf:
            print(f"DEBUG: Setting initial qpos: {self.robot_conf['init_qpos']}")
            self.robot.set_qpos(self.robot_conf["init_qpos"])
        else:
            print("DEBUG: No initial qpos provided in robot configuration")
            
        current_pose = self.get_end_effector_pose()
        self.target_pose = np.concatenate([current_pose.p, current_pose.q])
        print(f"DEBUG: Initial target pose: {self.target_pose}")
        
        # Initialize the joint parameters for the PID control
        self.robot_active_joints = self.robot.get_active_joints()
        print(f"DEBUG: Active joints: {[joint.name for joint in self.robot_active_joints]}")
        
        for i in range(len(self.robot_active_joints)):
            joint = self.robot_active_joints[i]
            joint.set_drive_property(stiffness=2000, damping=800, force_limit=100)
            if "init_qpos" in self.robot_conf:
                joint.set_drive_target(self.robot_conf["init_qpos"][i])
                
        # Use IK mode for our simple robot
        self.control_mode = "IK"
        print("DEBUG: Using IK control mode")
        self.IK_solver = self.robot.create_pinocchio_model()

    def robot_stop(self):
        self.ROBOT_STOP_FLAG = True

    def robot_reset(self):
        self.gripper_state = 1
        for i in range(len(self.robot_active_joints)):
            joint = self.robot_active_joints[i]
            joint.set_drive_target(self.robot_conf["init_qpos"][i])

    # Function to manipulate the robot end effector
    def robot_move_to_pose(self, pose):
        # The pose is for the moving group and relative to the root link
        if self.control_mode == "mplib":
            result = self.robot_planner.plan_screw(
                pose,
                self.robot.get_qpos(),
                time_step=self.timestep,
                use_point_cloud=self.use_point_cloud,
                use_attach=self.use_attach,
            )
            if result["status"] != "Success":
                result = self.robot_planner.plan(
                    pose,
                    self.robot.get_qpos(),
                    time_step=self.timestep,
                    use_point_cloud=self.use_point_cloud,
                    use_attach=self.use_attach,
                )
                if result["status"] != "Success":
                    print(result["status"])
                    return -1
        elif self.control_mode == "IK":
            # Get the gripper link index
            gripper_link_idx = next(i for i, link in enumerate(self.robot.get_links()) if link.name == "gripper_link")
            print(f"DEBUG: Target pose: position={pose[:3]}, quaternion={pose[3:]}")
            
            # Check if target is within reasonable workspace
            target_distance = np.linalg.norm(pose[:3])
            max_reach = 0.4  # Approximate maximum reach of the robot
            if target_distance > max_reach:
                print(f"WARNING: Target position {pose[:3]} is likely outside robot workspace (distance={target_distance:.3f}m, max_reach={max_reach:.3f}m)")
                return -1
            
            try:
                print(f"DEBUG: Current robot qpos={self.robot.get_qpos()}")
                print(f"DEBUG: Robot joint limits={self.robot.get_qlimits()}")
                
                inverse_kinematics = self.IK_solver.compute_inverse_kinematics(
                    link_index=gripper_link_idx,
                    pose=sapien.Pose(pose[:3], pose[3:]),
                    initial_qpos=self.robot.get_qpos(),
                )[0]
                print(f"DEBUG: Raw IK solution={inverse_kinematics}")
                
                # Try to optimize the IK solution
                optimized_qpos = self._optimize_IK(inverse_kinematics)
                print(f"DEBUG: Optimized IK solution={optimized_qpos}")
                
                # Verify the solution is within limits
                qlimits = self.robot.get_qlimits()
                for i, (qpos, (qmin, qmax)) in enumerate(zip(optimized_qpos, qlimits)):
                    if qpos < qmin or qpos > qmax:
                        print(f"ERROR: Joint {i} position {qpos:.4f} exceeds limits [{qmin:.4f}, {qmax:.4f}]")
                        return -1
                
                result = {}
                result["position"] = [optimized_qpos]
                return result
                
            except Exception as e:
                print(f"ERROR: IK computation failed: {str(e)}")
                return -1

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def _optimize_IK(self, qpos):
        """Optimize IK solution to respect joint limits"""
        qlimits = self.robot.get_qlimits()
        tolerance = 1e-6  # Small tolerance for numerical precision
        
        for i in range(len(qpos)):
            qpos[i] = self.normalize_angle(qpos[i])
            
            # Try to bring the angle within limits by adding/subtracting 2π
            while qpos[i] < qlimits[i][0] - tolerance:
                qpos[i] += 2 * np.pi
            while qpos[i] > qlimits[i][1] + tolerance:
                qpos[i] -= 2 * np.pi
                
            # Check if we succeeded in bringing the angle within limits
            if qpos[i] < qlimits[i][0] - tolerance or qpos[i] > qlimits[i][1] + tolerance:
                print(f"WARNING: Joint {i} angle {qpos[i]} cannot be brought within limits [{qlimits[i][0]}, {qlimits[i][1]}]")
                # Clamp to the nearest limit
                qpos[i] = np.clip(qpos[i], qlimits[i][0], qlimits[i][1])
                
        return qpos

    def robot_open_gripper(self):
        # Our simple robot doesn't have a gripper mechanism
        pass

    def robot_close_gripper(self):
        # Our simple robot doesn't have a gripper mechanism
        pass

    def gripper_fully_close(self):
        # Our simple robot doesn't have a gripper mechanism
        return True

    # Collision avoidence
    # Function to update the points for the robot to avoid collision
    def robot_add_point_cloud(self, points):
        if not self.use_point_cloud:
            self.use_point_cloud = True
        # The points should be a numpy array, in the robot root link coordinate
        self.robot_planner.update_point_cloud(points)

    # Function to add attach box to the robot to avoid collision
    def robot_add_attach(self, size, pose):
        if not self.use_attach:
            self.use_attach = True
        # The pose is related to the attached link, the default -1 is the moving group
        self.robot_planner.update_attached_box(size=size, pose=pose, link_id=-1)

    def _validate_target_pose(self, target_pose):
        """
        Validate if a target pose is within the robot's workspace.
        Returns (is_valid, message)
        """
        # Check input length
        if len(target_pose) != 7:
            return False, f"Target pose must have 7 values (got {len(target_pose)})"
            
        # Extract position and orientation
        position = target_pose[:3]
        quat = target_pose[3:]
        
        # Check position limits
        max_reach = 0.4  # Maximum reach of the robot arm
        distance = np.linalg.norm(position)
        if distance > max_reach:
            return False, f"Target position {position} is outside workspace (distance={distance:.3f}m, max={max_reach:.3f}m)"
            
        # Check height limits
        min_height = 0.0
        max_height = 0.3
        if position[2] < min_height or position[2] > max_height:
            return False, f"Target height {position[2]:.3f}m is outside limits ({min_height:.1f}m to {max_height:.1f}m)"
            
        # Check quaternion normalization
        quat_norm = np.linalg.norm(quat)
        if not np.isclose(quat_norm, 1.0, atol=1e-3):
            return False, f"Quaternion {quat} is not normalized (norm={quat_norm:.3f})"
            
        return True, "Target pose is valid"
