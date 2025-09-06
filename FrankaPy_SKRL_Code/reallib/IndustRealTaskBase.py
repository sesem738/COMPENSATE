# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Class definition for IndustRealTaskBase.

This script defines the IndustRealTaskBase base class. The class defines
the variables and methods that are common for all tasks.
"""

# Standard Library
import json
import os
import random

# Third Party
import numpy as np
import rospy
import torch
import yaml
from autolab_core import RigidTransform
from franka_interface_msgs.msg import SensorDataGroup
from frankapy import FrankaArm, FrankaConstants
from gym.spaces import Box
from rl_games.algos_torch.players import PpoPlayerContinuous
from scipy.spatial.transform import Rotation

# NVIDIA
import industreallib.control.scripts.control_utils as control_utils
import industreallib.perception.scripts.detect_objects as detect_objects
import industreallib.perception.scripts.map_workspace as map_workspace
import industreallib.perception.scripts.perception_utils as perception_utils


class IndustRealTaskBase:
    """Defines base class for all tasks."""

    def __init__(self, args, task_instance_config, in_sequence):
        """Initializes the configuration, goals, and robot for the task."""
        self._args = args
        self.task_instance_config = task_instance_config
        self._home_joint_angles = [
            0.0,
            -1.76076077e-01,
            0.0,
            -1.86691416e00,
            0.0,
            1.69344379e00,
            np.pi / 4,
        ]
        self._policy = None
        self.goal_coords = []
        self.goal_labels = []
        self._ros_publisher = None
        self._ros_rate = None
        self._ros_msg_count = 0
        self._device = "cuda"

        # For PLAI
        self._prev_targ_pos = None
        self._prev_targ_ori_mat = None

        # If task is not part of sequence, instantiate FrankaArm and get goals
        if not in_sequence:
            self.franka_arm = FrankaArm()
            control_utils.set_sigint_response(franka_arm=self.franka_arm)
            self._get_goals()

    def _go_to_goal_with_rl(self, goal, franka_arm):
        # Take fixed goal & add randomness to it
        # Update _get_observation   # Implementation pending
        # Update _get_actions   # Done, not tested, pending config params
        # Update _send_targets -> control_utils.compose_ros_msg    # Done, not tested
        # Update control_utils.get_pose_error      # Implementation pending, not tested, pending config params

        """Goes to a goal using an RL policy."""
        goal_pos = goal[:3]
        goal_ori_mat = Rotation.from_euler("XYZ", goal[3:]).as_matrix()  # intrinsic rotations

        self._start_target_stream(franka_arm=franka_arm)
        print("\nStarted streaming targets.")

        print("\nGoing to goal pose with RL...")
        # Get observations, get actions, send targets, and repeat
        initial_time = rospy.get_time()
        while rospy.get_time() - initial_time < self.task_instance_config.motion.duration:
            observations, curr_state = self._get_observations(
                goal_pos=goal_pos, goal_ori_mat=goal_ori_mat, franka_arm=franka_arm
            )
            actions = self._get_actions(observations=observations)
            self._send_targets(
                actions=actions,
                curr_pos=curr_state["pose"].translation,
                curr_ori_mat=curr_state["pose"].rotation,
            )

            # If current pose is close enough to goal pose, terminate early
            pos_err, ori_err_rad = control_utils.get_pose_error(
                curr_pos=curr_state["pose"].translation,
                curr_ori_mat=curr_state["pose"].rotation,
                targ_pos=goal_pos,
                targ_ori_mat=goal_ori_mat,
            )
            if (
                pos_err < self.task_instance_config.rl.pos_err_thresh
                and ori_err_rad < self.task_instance_config.rl.ori_err_rad_thresh
            ):
                print("Terminated early due to error below threshold.")
                break

            self._ros_rate.sleep()
        print("Finished going to goal pose with RL.")

        franka_arm.stop_skill()
        print("\nStopped streaming targets.")

        self._prev_targ_pos, self._prev_targ_ori_mat = None, None

        if self._args.debug_mode:
            control_utils.print_pose_error(
                curr_pos=curr_state["pose"].translation,
                curr_ori_mat=curr_state["pose"].rotation,
                targ_pos=goal_pos,
                targ_ori_mat=goal_ori_mat,
            )

    def _get_policy(self):
        """Gets an RL policy from rl-games."""
        # NOTE: Only PPO policies are currently supported.

        print("\nLoading an RL policy...")

        # Load config.yaml used in training
        with open(
            os.path.join(os.path.dirname(__file__), '..', '..', 'rl', 'checkpoints',
            self.task_instance_config.rl.checkpoint_name, 'config.yaml'),
            "r",
        ) as f:
            sim_config = yaml.safe_load(f)

        # Define env_info dict
        # NOTE: If not defined, rl-games will call rl_games.common.player.create_env() and
        # rl_games.common.env_configurations.get_env_info(). Afterward, rl-games will query
        # num_observations and num_actions. We only need to support those queries.
        # See rl_games.common.player.__init__() for more details.
        env_info = {
            "observation_space": Box(
                low=-np.Inf,
                high=np.Inf,
                shape=(sim_config["task"]["env"]["numObservations"],),
                dtype=np.float32,
            ),
            "action_space": Box(
                low=-1.0,
                high=1.0,
                shape=(sim_config["task"]["env"]["numActions"],),
                dtype=np.float32,
            ),
        }
        sim_config["train"]["params"]["config"]["env_info"] = env_info

        # Select device
        sim_config["train"]["params"]["config"]["device_name"] = self._device

        # Create rl-games agent
        policy = PpoPlayerContinuous(params=sim_config["train"]["params"])

        # Restore policy from checkpoint
        policy.restore(
            fn=(
                os.path.join(os.path.dirname(__file__), '..', '..', 'rl', 'checkpoints',
                self.task_instance_config.rl.checkpoint_name, 'nn',
                f"{self.task_instance_config.rl.checkpoint_name}.pth")
            )
        )

        # If RNN policy, reset RNN states
        policy.reset()

        print("Finished loading an RL policy.")

        return policy

    def _start_target_stream(self, franka_arm):
        """Starts streaming targets to franka-interface via frankapy."""
        self._ros_rate = rospy.Rate(self.task_instance_config.rl.policy_eval_freq)
        self._ros_publisher = rospy.Publisher(
            FrankaConstants.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )

        # Initiate streaming with dummy command to go to current joints
        franka_arm.goto_joints(
            joints=franka_arm.get_joints(),
            duration=5.0,
            dynamic=True,
            buffer_time=10.0,
            ignore_virtual_walls=False,
        )

    def _get_observations(self):
        """Gets observations from frankapy. Should be defined in a task-specific subclass."""
        raise NotImplementedError

    def _get_actions(self, observations):
        """Gets actions from the policy. Applies action scaling factors."""

        actions = self._policy.get_action(obs=observations, is_deterministic=True)
        actions *= self.dt * self._action_scale             #TODO: Get from environment / config
        actions = actions.detach().cpu().numpy()
        return actions

    def _send_targets(self, actions, curr_pos, curr_ori_mat):
        """Sends pose targets to franka-interface via frankapy."""
        # action: Delta joint angle (normalized)
        # curr_pos: Current joint angles

        if self.task_instance_config.control.mode.type == "nominal":
            #TODO: Ensure correct dimensions
            targ_pos = curr_pos + actions

        else:
            raise ValueError("Invalid control mode.")

        ros_msg = control_utils.compose_ros_msg(
            targ_pos=targ_pos,
            msg_count=self._ros_msg_count,
        )

        self._ros_publisher.publish(ros_msg)
        self._ros_msg_count += 1
