import os
import numpy as np
import torch
from isaacgymenvs.utils.joint_failure import JointFailure

from isaacgym import gymtorch, gymapi

# isaacgymenvs (VecTask class)
import sys
import isaacgymenvs
sys.path.append(list(isaacgymenvs.__path__)[0])
from tasks.base.vec_task import VecTask


class FrankaReach(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture = False, force_render = False):

        self.cfg = cfg
        # self.dt = 1 / 120.0 # set in create sim from sim params

        self._action_scale = self.cfg["env"]["actionScale"]
        self._dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self._control_space = self.cfg["env"]["controlSpace"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]  # name required for VecTask
        self.max_timestep = 500 # self.cfg["env"]["timesteps"]  #TODO:Arbitrary for now

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

       # Configure observation and action spaces
        self.cfg["env"]["numObservations"] = 18  # Actor observations
        self.cfg["env"]["numStates"] = 18 + 1  # Critic states (18 obs + 1 failed joint)

        if self._control_space == "joint":
            self.cfg["env"]["numActions"] = 7
        elif self._control_space == "cartesian":
            self.cfg["env"]["numActions"] = 3
        else:
            raise ValueError("Invalid control space: {}".format(self._control_space))
        
        self.useCurriculum = self.cfg["env"]["enableCurriculum"]
        if self.useCurriculum:
            self.curriculum_config = self.cfg["env"]["curriculum"]
            self.curriculum_switch_ratio = np.array(
                [ i/len(self.curriculum_config['joints']) for i in range(len(self.curriculum_config['joints'])) ]
            )
        else:
            self.curriculum_config = None

        self._end_effector_link = "panda_leftfinger"
        # self._end_effector_link = "fr3_leftfinger"

        # setup VecTask
        super().__init__(config=self.cfg,
                         rl_device=rl_device,
                         sim_device=sim_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless,
                         virtual_screen_capture=virtual_screen_capture,
                         force_render=force_render)

        # tensors and views: DOFs, roots, rigid bodies
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.root_state = gymtorch.wrap_tensor(self.root_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]

        self.root_pos = self.root_state[:, 0:3].view(self.num_envs, -1, 3)
        self.root_rot = self.root_state[:, 3:7].view(self.num_envs, -1, 4)
        self.root_vel_lin = self.root_state[:, 7:10].view(self.num_envs, -1, 3)
        self.root_vel_ang = self.root_state[:, 10:13].view(self.num_envs, -1, 3)

        self.rigid_body_pos = self.rigid_body_state[:, 0:3].view(self.num_envs, -1, 3)
        self.rigid_body_rot = self.rigid_body_state[:, 3:7].view(self.num_envs, -1, 4)
        self.rigid_body_vel_lin = self.rigid_body_state[:, 7:10].view(self.num_envs, -1, 3)
        self.rigid_body_vel_ang = self.rigid_body_state[:, 10:13].view(self.num_envs, -1, 3)

        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)

        # tensors and views: jacobian
        if self._control_space == "cartesian":
            jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot")
            self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)
            self.jacobian_end_effector = self.jacobian[:, self.rigid_body_dict_robot[self._end_effector_link] - 1, :, :7]

        # Testing single joint failure
        self.joint_failure = JointFailure(failure_type=self.curriculum_config['failure'], 
                                          dof_id_list=[self.curriculum_config['joints'][0]].copy(),
                                          failure_prob=0.25, 
                                          num_envs=self.num_envs, 
                                          max_ep_len=self.max_episode_length)
        
        self.timestep = 0
        # self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(isaacgymenvs.__file__)), "../assets")
        robot_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        # robot_asset_file = "urdf/franka_description/robots/fr3_franka_hand.urdf"

        # robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        # asset_options.replace_cylinder_with_capsule = False
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        # target asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.use_mesh_materials = True
        target_asset = self.gym.create_sphere(self.sim, 0.025, asset_options)

        robot_dof_stiffness = torch.tensor([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float32, device=self.device)
        robot_dof_damping = torch.tensor([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self.robot_dof_speed_limits = []
        for i in range(9):
            robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                robot_dof_props["stiffness"][i] = robot_dof_stiffness[i]
                robot_dof_props["damping"][i] = robot_dof_damping[i]
            else:
                robot_dof_props["stiffness"][i] = 7000.0
                robot_dof_props["damping"][i] = 50.0

            self.robot_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.robot_dof_upper_limits.append(robot_dof_props["upper"][i])
            self.robot_dof_speed_limits.append(robot_dof_props["velocity"][i])

        self.robot_dof_lower_limits = torch.tensor(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = torch.tensor(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_limits = torch.tensor(self.robot_dof_speed_limits, device=self.device)
        robot_dof_props["effort"][7] = 200
        robot_dof_props["effort"][8] = 200

        self.handle_targets = []
        self.handle_robots = []
        self.handle_envs = []

        indexes_sim_robot = []
        indexes_sim_target = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # create robot instance
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

            robot_actor = self.gym.create_actor(env=env_ptr,
                                                asset=robot_asset,
                                                pose=pose,
                                                name="robot",
                                                group=i, # collision group
                                                filter=1, # mask off collision
                                                segmentationId=0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            indexes_sim_robot.append(self.gym.get_actor_index(env_ptr, robot_actor, gymapi.DOMAIN_SIM))

            # create target instance
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.5, 0.0, 0.2)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)

            target_actor = self.gym.create_actor(env=env_ptr,
                                                 asset=target_asset,
                                                 pose=pose,
                                                 name="target",
                                                 group=i + 1, # collision group
                                                 filter=1, # mask off collision
                                                 segmentationId=1)
            indexes_sim_target.append(self.gym.get_actor_index(env_ptr, target_actor, gymapi.DOMAIN_SIM))

            self.gym.set_rigid_body_color(env_ptr, target_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1., 0., 0.))

            self.handle_envs.append(env_ptr)
            self.handle_robots.append(robot_actor)
            self.handle_targets.append(target_actor)

        self.indexes_sim_robot = torch.tensor(indexes_sim_robot, dtype=torch.int32, device=self.device)
        self.indexes_sim_target = torch.tensor(indexes_sim_target, dtype=torch.int32, device=self.device)

        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.rigid_body_dict_robot = self.gym.get_asset_rigid_body_dict(robot_asset)

        self.init_data()

    def init_data(self):
        self.robot_default_dof_pos = torch.tensor(np.radians([0, -45, 0, -135, 0, 90, 45, 0, 0]), device=self.device, dtype=torch.float32)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.num_robot_dofs), device=self.device, dtype=torch.float32)

        if self._control_space == "cartesian":
            self.end_effector_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.end_effector_rot = torch.zeros((self.num_envs, 4), device=self.device)

    def compute_reward(self):

        # If predicted failed joint id is == actual failed joint id, reward
        pred_reward = torch.where(self.pred_fail_joint == self.joint_failure.get_env_joint_failure_indices(), 0.0, -0.0)
        # print(f"Prediction Reward: {pred_reward}")
        # print(self.pred_fail_joint, self.joint_failure.get_env_joint_failure_indices())
        # print(self.joint_failure.get_env_joint_failure_indices())

        self.rew_buf[:] = -self._computed_distance + pred_reward

        self.reset_buf.fill_(0)
        # target reached
        self.reset_buf = torch.where(self._computed_distance <= 0.035, torch.ones_like(self.reset_buf), self.reset_buf)
        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # double restart correction (why?, is it necessary?)
        self.rew_buf = torch.where(self.progress_buf == 0, -0.75 * torch.ones_like(self.reset_buf), self.rew_buf)
        self.reset_buf = torch.where(self.progress_buf == 0, torch.zeros_like(self.reset_buf), self.reset_buf)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self._control_space == "cartesian":
            self.gym.refresh_jacobian_tensors(self.sim)

        robot_dof_pos = self.dof_pos
        robot_dof_vel = self.dof_vel
        self.end_effector_pos = self.rigid_body_pos[:, self.rigid_body_dict_robot[self._end_effector_link]]
        self.end_effector_rot = self.rigid_body_rot[:, self.rigid_body_dict_robot[self._end_effector_link]]
        target_pos = self.root_pos[:, 1]
        target_rot = self.root_rot[:, 1]

        dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) \
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0 \
            + (2* torch.rand((robot_dof_pos.shape[0],9), device=self.device) - 1) * 0.01
        
        # dof_vel_scaled = robot_dof_vel * self._dof_vel_scale
        dof_vel_scaled = robot_dof_vel / self.robot_dof_speed_limits \
            + (2* torch.rand((robot_dof_vel.shape[0],9), device=self.device) - 1) * 0.01

        # generalization_noise = torch.rand((dof_vel_scaled.shape[0], 7), device=self.device) + 0.5

        # Actor Observations (18)
        self.obs_buf[:, 0] = self.progress_buf / self.max_episode_length
        self.obs_buf[:, 1:8] = dof_pos_scaled[:, :7]
        self.obs_buf[:, 8:15] = dof_vel_scaled[:, :7] # * generalization_noise
        self.obs_buf[:, 15:18] = target_pos

        # Additional critic states (26 total)
        self.states_buf[:, :18] = self.obs_buf  # Copy actor observations
        self.states_buf[:, 18] = self.joint_failure.get_env_joint_failure_indices().float()  # Failed joint ID

        # compute distance for compute_reward()
        self._computed_distance = torch.norm(self.end_effector_pos - target_pos, dim=-1)

    def reset_idx(self, env_ids):

        # Apply curriculum
        if self.max_timestep > 0:
            completion_ratio = self.timestep / self.max_timestep
            completion_check = np.where(completion_ratio < self.curriculum_switch_ratio)[0]
            if len(completion_check) > 0:
                self.joint_failure.set_dof_id_list(dof_id_list=self.curriculum_config['joints'][:np.min(completion_check)].copy())
            else:
                self.joint_failure.set_dof_id_list(dof_id_list=self.curriculum_config['joints'].copy())

        # reset robot
        pos = torch.clamp(self.robot_default_dof_pos.unsqueeze(0) + 1 * (torch.rand((len(env_ids), self.num_robot_dofs), device=self.device) - 0.5),
                          self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        pos[:, 7:] = 0

        self.robot_dof_targets[env_ids, :] = pos[:]
        self.dof_pos[env_ids, :] = pos[:]
        self.dof_vel[env_ids, :] = 0

        indexes = self.indexes_sim_robot[env_ids]
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.robot_dof_targets),
                                                        gymtorch.unwrap_tensor(indexes),
                                                        len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(indexes),
                                              len(env_ids))

        # reset targets
        pos = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2
        pos[:, 0] = 0.50 + pos[:, 0] * 0.25
        pos[:, 1] = 0.00 + pos[:, 1] * 0.25
        pos[:, 2] = 0.20 + pos[:, 2] * 0.10

        self.root_pos[env_ids, 1, :] = pos[:]

        indexes = self.indexes_sim_target[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(indexes),
                                                     len(env_ids))

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # Reset joint failure info
        self.joint_failure.reset(env_ids=env_ids)
        

        # Find a better place for the viwer code
        if self.viewer:
            self.gym.viewer_camera_look_at(
                self.viewer, None, gymapi.Vec3(2, 2, 2), gymapi.Vec3(0,0,0))

    def pre_physics_step(self, actions):

        # Resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(reset_env_ids) > 0:
            actor_indices = self.reset_idx(reset_env_ids)


        actions = actions.clone().to(self.device)

        # if self._control_space == "joint":
        targets = self.robot_dof_targets[:, :7] + self.robot_dof_speed_scales[:7] * self.dt * actions * self._action_scale

        # elif self._control_space == "cartesian":
        #     goal_position = self.end_effector_pos + actions / 100.0
        #     delta_dof_pos = isaacgym_utils.ik(jacobian_end_effector=self.jacobian_end_effector,
        #                                       current_position=self.end_effector_pos,
        #                                       current_orientation=self.end_effector_rot,
        #                                       goal_position=goal_position,
        #                                       goal_orientation=None)
        #     targets = self.robot_dof_targets[:, :7] + delta_dof_pos

        self.joint_failure.apply(current_dofs=self.dof_pos, targets_dofs=targets, current_step=self.progress_buf)
        # print('Failed joint: ' + str(self.joint_failure.get_env_joint_failure_indices()))

        self.pred_fail_joint = torch.floor(3.99 * (actions[:, -1] + 1))  # maps [-1,1] to [0,7]
        
        self.robot_dof_targets[:, :7] = torch.clamp(targets, self.robot_dof_lower_limits[:7], self.robot_dof_upper_limits[:7])
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        self.timestep += 1
        
        self.compute_observations()
        self.compute_reward()