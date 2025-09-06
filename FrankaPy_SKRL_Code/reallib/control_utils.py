# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Control utilities module.

This module defines utility functions for controlling a Franka robot with
the frankapy library.
"""

# Standard Library
import random
import signal

# Third Party
import numpy as np
import rospy
from autolab_core import RigidTransform
from frankapy import SensorDataMessageType
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
from scipy.spatial.transform import Rotation

# Updated compose_ros_msg for joint position control
# Update get_pose_error to determine success/failure

def compose_ros_msg(targ_pos, msg_count):
    """Composes a ROS message to send to franka-interface for task-space joint angle control."""
    # NOTE: The sensor message classes expect the input joint positions to be represented as
    # (j0, j1, j2, j3, ...).

    curr_time = rospy.Time.now().to_time()
    traj_gen_proto_msg = JointPositionSensorMessage(
            id=msg_count, timestamp=curr_time, 
            joints=targ_pos
        )
    ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        )

    return ros_msg