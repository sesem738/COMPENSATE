#!/usr/bin/env python3
"""
Mock version of franky Robot and Gripper classes for testing without hardware.
"""

import numpy as np
import time
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Translation:
    """Mock translation class."""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError(f"Translation index {index} out of range")


@dataclass
class Rotation:
    """Mock rotation class."""
    def __init__(self, qw: float, qx: float, qy: float, qz: float):
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz


@dataclass
class Pose:
    """Mock pose class."""
    def __init__(self, translation: Translation, rotation: Rotation):
        self.translation = translation
        self.rotation = rotation


@dataclass
class EndEffectorPose:
    """Mock end effector pose."""
    def __init__(self, translation: Translation, rotation: Rotation):
        self.translation = translation
        self.rotation = rotation


@dataclass
class CartesianPose:
    """Mock cartesian pose."""
    def __init__(self, end_effector_pose: EndEffectorPose):
        self.end_effector_pose = end_effector_pose


@dataclass
class CartesianState:
    """Mock cartesian state."""
    def __init__(self, pose: CartesianPose):
        self.pose = pose


@dataclass
class JointState:
    """Mock joint state."""
    def __init__(self, position: List[float], velocity: List[float]):
        self.position = position
        self.velocity = velocity


class JointMotion:
    """Mock joint motion class."""
    def __init__(self, target_joints: np.ndarray):
        self.target_joints = np.array(target_joints)


class MockRobot:
    """Mock Robot class that simulates Franka robot behavior."""
    
    def __init__(self, ip: str):
        self.ip = ip
        print(f"Mock: Connecting to robot at {ip}...")
        time.sleep(0.5)  # Simulate connection delay
        
        # Robot state
        self.relative_dynamics_factor = 0.15
        self._joint_positions = np.array([0, -0.569, 0, -2.81, 0, 3.037, 0.741])  # Default pose
        self._joint_velocities = np.zeros(7)
        
        # End effector position (computed from forward kinematics approximation)
        self._end_effector_pos = np.array([0.5, 0.0, 0.3])  # Starting position
        
        print("Mock: Robot connected")
    
    def _compute_forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Simplified forward kinematics approximation.
        In reality, this would be complex FK calculation.
        """
        # Simple approximation based on joint changes
        # This is not accurate FK, just for testing purposes
        base_pos = np.array([0.5, 0.0, 0.3])
        
        # Add some movement based on joint positions
        delta = np.array([
            0.3 * np.sin(joint_positions[0]) + 0.2 * np.cos(joint_positions[1]),
            0.3 * np.cos(joint_positions[0]) + 0.2 * np.sin(joint_positions[2]),
            0.2 * joint_positions[1] + 0.1 * joint_positions[3]
        ])
        
        position = base_pos + delta
        # Keep within reasonable workspace
        position = np.clip(position, [0.2, -0.5, 0.1], [0.8, 0.5, 0.8])
        
        return position
    
    @property
    def current_joint_state(self) -> JointState:
        """Get current joint state."""
        return JointState(
            position=self._joint_positions.tolist(),
            velocity=self._joint_velocities.tolist()
        )
    
    @property
    def current_cartesian_state(self) -> CartesianState:
        """Get current cartesian state."""
        # Update end effector position based on current joints
        self._end_effector_pos = self._compute_forward_kinematics(self._joint_positions)
        
        translation = Translation(
            self._end_effector_pos[0],
            self._end_effector_pos[1], 
            self._end_effector_pos[2]
        )
        rotation = Rotation(0.0, 0.0, 0.0, 1.0)  # Identity quaternion
        end_effector_pose = EndEffectorPose(translation, rotation)
        cartesian_pose = CartesianPose(end_effector_pose)
        
        return CartesianState(cartesian_pose)
    
    def move(self, motion: JointMotion, asynchronous: bool = False):
        """Execute a joint motion."""
        target_joints = motion.target_joints
        
        # Simulate motion by interpolating to target
        steps = 10
        for step in range(steps):
            alpha = (step + 1) / steps
            self._joint_positions = (1 - alpha) * self._joint_positions + alpha * target_joints
            
            # Simulate joint velocities during motion
            if step < steps - 1:
                self._joint_velocities = (target_joints - self._joint_positions) / (self.relative_dynamics_factor * 0.1)
            else:
                self._joint_velocities = np.zeros(7)  # Stop at target
            
            if not asynchronous:
                time.sleep(0.01)  # Simulate motion time
        
        # Clamp to joint limits
        joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        self._joint_positions = np.clip(self._joint_positions, joint_limits_lower, joint_limits_upper)
        
        # Add small amount of noise to simulate real robot behavior
        self._joint_positions += np.random.normal(0, 0.001, 7)


class MockGripper:
    """Mock Gripper class."""
    
    def __init__(self, ip: str):
        self.ip = ip
        self.width = 0.08  # Default open width
        print(f"Mock: Gripper connected at {ip}")
    
    def move(self, width: float, speed: float = 0.1):
        """Move gripper to specified width."""
        print(f"Mock: Moving gripper to width={width:.3f} at speed={speed:.2f}")
        self.width = width
        time.sleep(0.2)  # Simulate gripper motion time


# Aliases to match the real franky interface
Robot = MockRobot
Gripper = MockGripper