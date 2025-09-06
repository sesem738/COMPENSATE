#!/usr/bin/env python3
"""
Test version of FrankaReachPose environment using mock robot interfaces.
This allows testing the environment logic without physical hardware.
"""

import gymnasium as gym
import time
import numpy as np
import torch
import sys
import os

# Import our mock robot instead of the real franky
from mock_franky import Robot, Gripper, JointMotion


class FrankaReachPoseTest(gym.Env):
    """Test version of FrankaReachPose using mock robot interfaces."""
    
    def __init__(self, robot_ip="172.16.0.2", device="cuda:0", auto_target=True):
        super().__init__()
        
        self.device = device
        self.robot_ip = robot_ip
        self.auto_target = auto_target  # Skip user input when True
        
        # Gym spaces - compatible with IsaacLab reach task
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(28,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-4, high=4, shape=(7,), dtype=np.float32)
        
        # Connect to mock robot
        print(f"Connecting to mock robot at {robot_ip}...")
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip) 
        print("Mock robot connected")
        
        # Safety settings
        self.robot.relative_dynamics_factor = 0.15  # 15% of max speed for safety
        
        # Control parameters
        self.dt = 1 / 30.0
        self.action_scale = 0.5  # Smaller scale for safety
        self.dof_vel_scale = 0.1
        self.max_episode_length = 50  # Shorter episodes for testing
        
        # Target and home positions
        self.target_pos = np.array([0.5, 0.0, 0.3])
        self.robot_default_dof_pos = np.array([0, -0.569, 0, -2.81, 0, 3.037, 0.741])
        self.previous_action = np.zeros(7)
        
        # Joint limits for safety
        self.robot_dof_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.robot_dof_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Workspace limits for safety
        self.workspace_limits = {
            'x': (0.3, 0.8),
            'y': (-0.4, 0.4), 
            'z': (0.3, 0.8)
        }
        
        # State tracking
        self.progress_buf = 0
        self.obs_buf = np.zeros((28,), dtype=np.float32)  # Fixed size to match observation_space
        self.episode_count = 0
        
    def _get_observation_reward_done(self):
        """Get current observation, reward, and done status."""
        # Get robot state
        cartesian_state = self.robot.current_cartesian_state
        joint_state = self.robot.current_joint_state
        
        robot_dof_pos = np.array(joint_state.position)
        robot_dof_vel = np.array(joint_state.velocity)
        
        # Get end effector position
        end_effector_pose = cartesian_state.pose.end_effector_pose
        end_effector_pos = np.array([
            end_effector_pose.translation[0],
            end_effector_pose.translation[1],
            end_effector_pose.translation[2]
        ])
        
        # Build observation (compatible with IsaacLab format)
        self.obs_buf[:7] = robot_dof_pos - self.robot_default_dof_pos
        self.obs_buf[7:14] = robot_dof_vel
        self.obs_buf[14:17] = self.target_pos  # Only position, not full pose
        self.obs_buf[17:24] = self.previous_action 
        self.obs_buf[24:27] = end_effector_pos  # Current end effector position
        self.obs_buf[27] = self.progress_buf / float(self.max_episode_length)  # Progress
        
        # Calculate reward (negative distance to target)
        distance = np.linalg.norm(end_effector_pos - self.target_pos)
        reward = -distance
        
        # Add success bonus
        if distance <= 0.03:  # 3cm tolerance
            reward += 10.0
            
        # Check termination conditions
        done = bool(self.progress_buf >= self.max_episode_length - 1)
        done = done or bool(distance <= 0.03)  # Success termination
        
        if done:
            success = distance <= 0.03
            print(f"Episode finished - Success: {success}, Distance: {distance:.4f}")
            
        return self.obs_buf.copy(), reward, done
        
    def _generate_random_target(self):
        """Generate a random target within workspace."""
        x_min, x_max = self.workspace_limits['x']
        y_min, y_max = self.workspace_limits['y']
        z_min, z_max = self.workspace_limits['z']
        
        # Generate random target with some bias towards reachable area
        self.target_pos = np.array([
            np.random.uniform(x_min + 0.1, x_max - 0.1),
            np.random.uniform(y_min + 0.1, y_max - 0.1),
            np.random.uniform(z_min + 0.1, z_max - 0.1)
        ])
        
    def reset(self, seed=None, options=None):
        self.episode_count += 1
        print(f"\n=== Test Episode {self.episode_count} Reset ===")

        # Move to home position
        print("Moving to home position...")
        home_motion = JointMotion(self.robot_default_dof_pos)
        self.robot.move(home_motion)
        self.gripper.move(width=0.08, speed=0.1)  # Open gripper
        
        # Add small random offset to starting position
        print("Moving to random start position...")
        start_offset = 0.1 * (np.random.rand(7) - 0.5)
        start_pos = self.robot_default_dof_pos + start_offset
        
        # Safety check for start position
        start_pos = np.clip(start_pos, 
                           self.robot_dof_lower_limits + 0.1,
                           self.robot_dof_upper_limits - 0.1)
        
        start_motion = JointMotion(start_pos)
        self.robot.move(start_motion)
        
        # Set target position
        if self.auto_target:
            self._generate_random_target()
            print(f"Auto-generated target position: {self.target_pos}")
        else:
            # get target position from prompt
            while True:
                try:
                    print("Enter target position (X, Y, Z) in meters")
                    raw = input("or press [Enter] key for a random target position: ")
                    if raw:
                        self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                    else:
                        self._generate_random_target()
                    print("Target position:", self.target_pos)
                    break
                except ValueError:
                    print("Invalid input. Try something like: 0.65, 0.0, 0.2")

            input("Press [Enter] to continue")
        
        # Reset episode state
        self.progress_buf = 0
        self.previous_action = np.zeros(7)
        
        # Get initial observation
        observation, _, _ = self._get_observation_reward_done()
        
        info = {
            'episode': self.episode_count,
            'target_position': self.target_pos.tolist(),
            'is_success': False
        }
        
        return observation, info
        
    def step(self, action):
        """Execute one step in the environment."""
        self.progress_buf += 1
        
        # Store previous action for observation
        self.previous_action = action.copy()
        
        # Get current joint positions
        current_state = self.robot.current_joint_state
        current_joints = np.array(current_state.position)
        
        # Calculate target joint positions
        target_joints = current_joints + (self.dt * action * self.action_scale)
        
        # Safety clipping
        target_joints = np.clip(target_joints, 
                               self.robot_dof_lower_limits + 0.1,
                               self.robot_dof_upper_limits - 0.1)
        
        # Execute motion
        motion = JointMotion(target_joints)
        
        try:
            self.robot.move(motion, asynchronous=True)
        except Exception as e:
            print(f"Motion failed: {e}")
            # Don't crash, just continue with current position
            
        # Control rate limiting (shorter for testing)
        time.sleep(self.dt * 0.1)  # 10x faster for testing
        
        # Get observation, reward, and done status
        observation, reward, terminated = self._get_observation_reward_done()
        truncated = False  # We use terminated for both success and timeout
        
        # Get end effector position for info
        current_ee_pos = np.array([
            self.robot.current_cartesian_state.pose.end_effector_pose.translation[0],
            self.robot.current_cartesian_state.pose.end_effector_pose.translation[1],
            self.robot.current_cartesian_state.pose.end_effector_pose.translation[2]
        ])
        
        # Info dictionary
        info = {
            'episode_step': self.progress_buf,
            'is_success': terminated and reward > 0,  # Success if terminated with positive reward
            'distance_to_target': np.linalg.norm(current_ee_pos - self.target_pos)
        }
        
        return observation, reward, terminated, truncated, info
        
    def render(self, mode='human'):
        """Render the environment (no-op for test robot)."""
        pass
        
    def close(self):
        """Clean up environment."""
        print("Closing test environment...")
        try:
            # Move to safe position before closing
            home_motion = JointMotion(self.robot_default_dof_pos)
            self.robot.move(home_motion)
            print("Mock robot returned to home position")
        except Exception as e:
            print(f"Warning: Could not return robot to home position: {e}")


# Create an alias that matches the real environment name for easy testing
FrankaReachPose = FrankaReachPoseTest