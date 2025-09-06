#!/usr/bin/env python3

"""
Real-world environment interface for Franka Emika robot using franky library.

This module provides a real-world alternative to IsaacLab simulation environments,
maintaining compatibility with the IsaacLab ManagerBasedRLEnv interface while
controlling a physical Franka Panda robot.
"""

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import time
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Franky library for real robot control
try:
    from franky import Robot, Gripper, JointMotion, CartesianMotion, ImpedanceMotion
except ImportError:
    print("Warning: franky library not available. Install with: pip install franky")
    Robot = Gripper = JointMotion = CartesianMotion = ImpedanceMotion = None

# IsaacLab compatibility (mock if not available)
try:
    from isaaclab.envs import ManagerBasedRLEnvCfg, VecEnvObs, VecEnvStepReturn
    ISAACLAB_AVAILABLE = True
except ImportError:
    print("Warning: IsaacLab not available. Using mock interface.")
    ISAACLAB_AVAILABLE = False
    ManagerBasedRLEnvCfg = object
    VecEnvObs = Dict[str, torch.Tensor]
    VecEnvStepReturn = Tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, Dict]


@dataclass
class RealWorldRobotState:
    """Container for real robot state information."""
    
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    end_effector_pose: np.ndarray  # [x, y, z, qx, qy, qz, qw]
    end_effector_position: np.ndarray  # [x, y, z]
    gripper_width: float
    timestamp: float
    is_moving: bool = False
    last_error: Optional[str] = None


@dataclass
class RealWorldEnvConfig:
    """Configuration for real-world environment interface."""
    
    # Robot connection
    robot_ip: str = "172.16.0.2"
    connect_gripper: bool = True
    
    # Control parameters
    control_mode: str = "joint_position"  # "joint_position", "cartesian", "impedance"
    control_frequency: float = 30.0  # Hz
    action_scale: float = 0.1
    max_joint_velocity: float = 2.0
    max_cartesian_velocity: float = 0.5
    
    # Safety parameters
    joint_limits_buffer: float = 0.1  # radians buffer from joint limits
    workspace_limits: Dict[str, Tuple[float, float]] = None
    collision_detection: bool = True
    emergency_stop_enabled: bool = True
    
    # Task parameters
    episode_length: float = 15.0  # seconds
    target_tolerance: float = 0.02  # meters
    orientation_tolerance: float = 0.1  # radians
    
    # Environment settings
    num_envs: int = 1  # Real robot = single environment
    observation_dim: int = 18  # Compatible with IsaacLab reach task
    action_dim: int = 7  # 7 DoF joint control
    
    def __post_init__(self):
        """Set default workspace limits if not provided."""
        if self.workspace_limits is None:
            self.workspace_limits = {
                "x": (0.3, 0.8),   # meters
                "y": (-0.4, 0.4),  # meters  
                "z": (0.1, 0.8),   # meters
            }


class RealWorldFrankaEnv:
    """
    Real-world Franka robot environment interface.
    
    Provides compatibility with IsaacLab ManagerBasedRLEnv while controlling
    a physical Franka Panda robot via the franky library.
    """
    
    def __init__(self, cfg: RealWorldEnvConfig):
        """Initialize the real-world environment.
        
        Args:
            cfg: Configuration for the real-world environment
        """
        self.cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Robot state
        self._robot_state = RealWorldRobotState(
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7), 
            end_effector_pose=np.zeros(7),
            end_effector_position=np.zeros(3),
            gripper_width=0.0,
            timestamp=time.time()
        )
        
        # Episode management
        self._episode_start_time = 0.0
        self._episode_step = 0
        self._max_episode_steps = int(cfg.episode_length * cfg.control_frequency)
        
        # Target management
        self._current_target = np.array([0.5, 0.0, 0.3])  # Default target position
        self._target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Default orientation (qw, qx, qy, qz)
        
        # Action and observation spaces (Gymnasium compatible)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.observation_dim,), dtype=np.float32
        )
        
        # Robot connection (lazy initialization)
        self._robot: Optional[Robot] = None
        self._gripper: Optional[Gripper] = None
        self._is_connected = False
        
        # Safety monitoring
        self._emergency_stopped = False
        self._last_action_time = 0.0
        
        print(f"Real-world Franka environment initialized")
        print(f"Control mode: {cfg.control_mode}")
        print(f"Control frequency: {cfg.control_frequency} Hz")
        print(f"Episode length: {cfg.episode_length} seconds ({self._max_episode_steps} steps)")
        
    @property 
    def device(self) -> torch.device:
        """Device for tensor operations."""
        return self._device
        
    @property
    def num_envs(self) -> int:
        """Number of environments (always 1 for real robot)."""
        return 1
        
    def _connect_robot(self) -> bool:
        """Connect to the real robot.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self._is_connected:
            return True
            
        try:
            print(f"Connecting to Franka robot at {self.cfg.robot_ip}...")
            self._robot = Robot(self.cfg.robot_ip)
            
            if self.cfg.connect_gripper:
                self._gripper = Gripper(self.cfg.robot_ip)
                
            # Set safety parameters
            self._robot.relative_dynamics_factor = 0.2  # 20% of max speed for safety
            
            self._is_connected = True
            print("✓ Robot connection established")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to robot: {e}")
            self._robot_state.last_error = str(e)
            return False
            
    def _disconnect_robot(self):
        """Safely disconnect from the robot."""
        if self._is_connected:
            try:
                # Return to safe home position
                self._move_to_home_position()
                print("✓ Robot disconnected safely")
            except Exception as e:
                print(f"Warning: Error during disconnect: {e}")
            finally:
                self._robot = None
                self._gripper = None  
                self._is_connected = False
                
    def _update_robot_state(self):
        """Update internal robot state from real robot."""
        if not self._is_connected or self._robot is None:
            return
            
        try:
            # Get joint state
            joint_state = self._robot.current_joint_state
            self._robot_state.joint_positions = np.array(joint_state.position)
            self._robot_state.joint_velocities = np.array(joint_state.velocity)
            
            # Get Cartesian state  
            cartesian_state = self._robot.current_cartesian_state
            ee_pose = cartesian_state.pose.end_effector_pose
            
            self._robot_state.end_effector_position = np.array([
                ee_pose.translation[0],
                ee_pose.translation[1],
                ee_pose.translation[2]
            ])
            
            self._robot_state.end_effector_pose = np.array([
                ee_pose.translation[0],
                ee_pose.translation[1], 
                ee_pose.translation[2],
                ee_pose.rotation[0],  # qx
                ee_pose.rotation[1],  # qy
                ee_pose.rotation[2],  # qz
                ee_pose.rotation[3],  # qw
            ])
            
            # Get gripper state if available
            if self._gripper is not None:
                gripper_state = self._gripper.current_state
                self._robot_state.gripper_width = gripper_state.width
            
            self._robot_state.timestamp = time.time()
            self._robot_state.is_moving = self._robot.current_cartesian_state.is_moving
            
        except Exception as e:
            print(f"Warning: Failed to update robot state: {e}")
            self._robot_state.last_error = str(e)
            
    def _check_safety_limits(self, target_joints: np.ndarray) -> bool:
        """Check if target joint positions are within safety limits.
        
        Args:
            target_joints: Target joint positions to check
            
        Returns:
            True if safe, False otherwise
        """
        # Standard Franka joint limits with safety buffer
        joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Add safety buffer
        joint_limits_lower += self.cfg.joint_limits_buffer
        joint_limits_upper -= self.cfg.joint_limits_buffer
        
        # Check limits
        if np.any(target_joints < joint_limits_lower) or np.any(target_joints > joint_limits_upper):
            print("Warning: Target joints outside safety limits")
            return False
            
        return True
        
    def _check_workspace_limits(self, target_position: np.ndarray) -> bool:
        """Check if target position is within workspace limits.
        
        Args:
            target_position: Target Cartesian position [x, y, z]
            
        Returns:
            True if within workspace, False otherwise
        """
        x, y, z = target_position
        
        x_min, x_max = self.cfg.workspace_limits["x"]
        y_min, y_max = self.cfg.workspace_limits["y"] 
        z_min, z_max = self.cfg.workspace_limits["z"]
        
        return (x_min <= x <= x_max and 
                y_min <= y <= y_max and
                z_min <= z <= z_max)
                
    def _move_to_home_position(self):
        """Move robot to safe home position."""
        if not self._is_connected or self._robot is None:
            return
            
        try:
            # Standard Franka home position
            home_joints = np.array([0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398])
            motion = JointMotion(home_joints)
            self._robot.move(motion)
            
            if self._gripper is not None:
                self._gripper.move(width=0.08, speed=0.1)  # Open gripper
                
        except Exception as e:
            print(f"Warning: Failed to move to home position: {e}")
            
    def _generate_random_target(self) -> np.ndarray:
        """Generate a random target position within workspace limits.
        
        Returns:
            Random target position [x, y, z]
        """
        x_min, x_max = self.cfg.workspace_limits["x"]
        y_min, y_max = self.cfg.workspace_limits["y"]
        z_min, z_max = self.cfg.workspace_limits["z"]
        
        target = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max), 
            np.random.uniform(z_min, z_max)
        ])
        
        return target
        
    def _get_observation(self) -> torch.Tensor:
        """Compute observation tensor compatible with IsaacLab reach task.
        
        Returns:
            Observation tensor of shape (1, obs_dim)
        """
        self._update_robot_state()
        
        # Build observation (compatible with IsaacLab reach task format)
        obs = np.zeros(self.cfg.observation_dim, dtype=np.float32)
        
        # Episode progress (normalized)
        obs[0] = self._episode_step / self._max_episode_steps
        
        # Joint positions (normalized to [-1, 1])
        joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        joint_pos_normalized = 2.0 * (self._robot_state.joint_positions - joint_limits_lower) / \
                              (joint_limits_upper - joint_limits_lower) - 1.0
        obs[1:8] = joint_pos_normalized
        
        # Joint velocities (scaled)
        obs[8:15] = self._robot_state.joint_velocities * 0.1
        
        # Target position
        obs[15:18] = self._current_target
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self._device)
        return obs_tensor.unsqueeze(0)  # Shape: (1, obs_dim)
        
    def _compute_reward(self) -> float:
        """Compute reward based on distance to target.
        
        Returns:
            Reward value
        """
        # Distance-based reward (negative distance)
        distance = np.linalg.norm(self._robot_state.end_effector_position - self._current_target)
        position_reward = -distance
        
        # Bonus for reaching target
        if distance < self.cfg.target_tolerance:
            position_reward += 10.0
            
        # Action smoothness penalty
        velocity_penalty = -0.001 * np.sum(np.square(self._robot_state.joint_velocities))
        
        total_reward = position_reward + velocity_penalty
        return float(total_reward)
        
    def _check_termination(self) -> Tuple[bool, bool]:
        """Check episode termination conditions.
        
        Returns:
            Tuple of (terminated, truncated)
        """
        # Success termination: reached target
        distance = np.linalg.norm(self._robot_state.end_effector_position - self._current_target)
        success = distance < self.cfg.target_tolerance
        
        # Time limit truncation
        time_limit = self._episode_step >= self._max_episode_steps
        
        # Safety termination
        safety_violation = (self._emergency_stopped or 
                           self._robot_state.last_error is not None)
        
        terminated = success or safety_violation
        truncated = time_limit
        
        return terminated, truncated
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[VecEnvObs, Dict]:
        """Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observations, info)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Connect to robot if not already connected
        if not self._connect_robot():
            raise RuntimeError("Failed to connect to robot")
            
        # Reset episode state
        self._episode_start_time = time.time()
        self._episode_step = 0
        self._emergency_stopped = False
        
        # Move to safe starting position
        print("Resetting robot to starting position...")
        self._move_to_home_position()
        
        # Add small random offset to starting position
        if self._robot is not None:
            current_joints = self._robot_state.joint_positions.copy()
            random_offset = 0.1 * (np.random.rand(7) - 0.5)  # ±5% random offset
            start_joints = current_joints + random_offset
            
            if self._check_safety_limits(start_joints):
                try:
                    motion = JointMotion(start_joints)
                    self._robot.move(motion)
                except Exception as e:
                    print(f"Warning: Failed to move to start position: {e}")
        
        # Generate new target
        self._current_target = self._generate_random_target()
        print(f"New target: {self._current_target}")
        
        # Get initial observation
        obs = self._get_observation()
        
        # Return in IsaacLab format
        obs_dict = {"policy": obs}
        info_dict = {
            "target_position": self._current_target.tolist(),
            "episode_step": self._episode_step,
            "robot_connected": self._is_connected,
        }
        
        return obs_dict, info_dict
        
    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        """Execute one environment step.
        
        Args:
            actions: Action tensor of shape (1, action_dim)
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        if not self._is_connected or self._robot is None:
            raise RuntimeError("Robot not connected")
            
        # Convert actions to numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy().squeeze()
            
        # Apply action based on control mode
        try:
            if self.cfg.control_mode == "joint_position":
                self._apply_joint_position_action(actions)
            elif self.cfg.control_mode == "cartesian":
                self._apply_cartesian_action(actions)
            elif self.cfg.control_mode == "impedance":
                self._apply_impedance_action(actions)
            else:
                raise ValueError(f"Unknown control mode: {self.cfg.control_mode}")
                
        except Exception as e:
            print(f"Error applying action: {e}")
            self._robot_state.last_error = str(e)
            
        # Wait for control period
        time.sleep(1.0 / self.cfg.control_frequency)
        
        # Update state and compute outputs
        self._episode_step += 1
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated, truncated = self._check_termination()
        
        # Create tensors
        obs_dict = {"policy": obs}
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self._device)
        terminated_tensor = torch.tensor([terminated], dtype=torch.bool, device=self._device)
        truncated_tensor = torch.tensor([truncated], dtype=torch.bool, device=self._device)
        
        # Info dictionary
        distance = np.linalg.norm(self._robot_state.end_effector_position - self._current_target)
        info_dict = {
            "distance_to_target": distance,
            "episode_step": self._episode_step,
            "success": distance < self.cfg.target_tolerance,
            "robot_state": {
                "joint_positions": self._robot_state.joint_positions.tolist(),
                "end_effector_position": self._robot_state.end_effector_position.tolist(),
                "is_moving": self._robot_state.is_moving,
            }
        }
        
        return obs_dict, reward_tensor, terminated_tensor, truncated_tensor, info_dict
        
    def _apply_joint_position_action(self, actions: np.ndarray):
        """Apply joint position control action.
        
        Args:
            actions: Joint position increments
        """
        current_joints = self._robot_state.joint_positions
        target_joints = current_joints + actions * self.cfg.action_scale
        
        if self._check_safety_limits(target_joints):
            motion = JointMotion(target_joints)
            self._robot.move(motion, asynchronous=True)
        else:
            print("Warning: Action would violate joint limits, skipping")
            
    def _apply_cartesian_action(self, actions: np.ndarray):
        """Apply Cartesian space action.
        
        Args:
            actions: Cartesian position/orientation increments
        """
        # Extract position and orientation components from action
        pos_action = actions[:3] * self.cfg.action_scale
        
        current_pos = self._robot_state.end_effector_position
        target_pos = current_pos + pos_action
        
        if self._check_workspace_limits(target_pos):
            # Create Cartesian motion (position only for simplicity)
            motion = CartesianMotion(target_pos.tolist())
            self._robot.move(motion, asynchronous=True)
        else:
            print("Warning: Action would violate workspace limits, skipping")
            
    def _apply_impedance_action(self, actions: np.ndarray):
        """Apply impedance control action.
        
        Args:
            actions: Desired force/torque or position increments
        """
        # Simplified impedance control - modify position with compliance
        pos_action = actions[:3] * self.cfg.action_scale * 0.5  # Reduced scale for safety
        
        current_pos = self._robot_state.end_effector_position  
        target_pos = current_pos + pos_action
        
        if self._check_workspace_limits(target_pos):
            # Use impedance motion with soft compliance
            motion = ImpedanceMotion(target_pos.tolist())
            self._robot.move(motion, asynchronous=True)
        else:
            print("Warning: Impedance action would violate workspace limits, skipping")
            
    def close(self):
        """Clean up environment resources."""
        print("Closing real-world environment...")
        self._disconnect_robot()
        print("✓ Environment closed")
        
    def __del__(self):
        """Destructor to ensure safe cleanup."""
        if hasattr(self, '_is_connected') and self._is_connected:
            self.close()


def create_real_world_env(robot_ip: str = "172.16.0.2", 
                         control_mode: str = "joint_position",
                         **kwargs) -> RealWorldFrankaEnv:
    """Factory function to create a real-world Franka environment.
    
    Args:
        robot_ip: IP address of the Franka robot
        control_mode: Control mode ("joint_position", "cartesian", "impedance")
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured real-world environment
    """
    cfg = RealWorldEnvConfig(
        robot_ip=robot_ip,
        control_mode=control_mode,
        **kwargs
    )
    
    return RealWorldFrankaEnv(cfg)


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = create_real_world_env(
        robot_ip="172.16.0.2",
        control_mode="joint_position", 
        episode_length=10.0,
        control_frequency=10.0  # Lower frequency for testing
    )
    
    try:
        print("Testing real-world environment...")
        
        # Reset environment
        obs, info = env.reset()
        print(f"Initial observation shape: {obs['policy'].shape}")
        print(f"Initial info: {info}")
        
        # Run a few steps
        for step in range(5):
            # Random action
            action = torch.randn(1, 7) * 0.1  # Small random actions
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step + 1}:")
            print(f"  Reward: {reward.item():.4f}")
            print(f"  Distance: {info['distance_to_target']:.4f}")
            print(f"  Terminated: {terminated.item()}")
            print(f"  Truncated: {truncated.item()}")
            
            if terminated.item() or truncated.item():
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        env.close()
        print("Test completed")