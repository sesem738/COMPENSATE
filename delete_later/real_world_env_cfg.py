#!/usr/bin/env python3

"""
Configuration system for real-world Franka robot environment.

This module provides IsaacLab-compatible configuration classes for the real-world
environment, enabling seamless integration with existing IsaacLab training pipelines.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import IsaacLab components for compatibility
try:
    from isaaclab.utils import configclass
    from isaaclab.envs import ManagerBasedRLEnvCfg
    ISAACLAB_AVAILABLE = True
except ImportError:
    # Fallback decorator and base class if IsaacLab not available
    def configclass(cls):
        return dataclass(cls)
    ManagerBasedRLEnvCfg = object
    ISAACLAB_AVAILABLE = False


@configclass
class RealWorldRobotCfg:
    """Configuration for real robot connection and control."""
    
    # Connection parameters
    robot_ip: str = "172.16.0.2"
    connect_gripper: bool = True
    connection_timeout: float = 10.0
    
    # Control parameters  
    control_mode: str = "joint_position"  # "joint_position", "cartesian", "impedance"
    control_frequency: float = 30.0  # Hz
    relative_dynamics_factor: float = 0.2  # Safety: 20% of max robot speed
    
    # Joint limits and safety
    joint_position_limits: Tuple[List[float], List[float]] = (
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],  # lower
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]         # upper
    )
    joint_velocity_limits: List[float] = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
    
    # Workspace limits [x_min, x_max, y_min, y_max, z_min, z_max]
    workspace_limits: List[float] = [0.25, 0.8, -0.4, 0.4, 0.1, 0.8]
    
    # Safety parameters
    joint_limits_buffer: float = 0.05  # radians safety buffer from joint limits
    collision_detection: bool = True
    emergency_stop_enabled: bool = True
    max_contact_force: float = 20.0  # Newtons


@configclass  
class RealWorldActionsCfg:
    """Configuration for action space and scaling."""
    
    # Action space configuration
    action_dim: int = 7  # 7-DOF joint control
    action_scale: float = 0.1  # Scale factor for actions
    use_normalized_actions: bool = True  # Actions in [-1, 1] range
    
    # Control-specific parameters
    joint_position_scale: float = 0.1    # radians per action unit
    cartesian_position_scale: float = 0.02  # meters per action unit  
    cartesian_orientation_scale: float = 0.1  # radians per action unit
    
    # Action smoothing and filtering
    action_filter_enabled: bool = True
    action_filter_alpha: float = 0.3  # Low-pass filter coefficient
    max_action_rate: float = 2.0  # Maximum action change per step


@configclass
class RealWorldObservationsCfg:
    """Configuration for observation space and processing."""
    
    # Observation dimensions
    observation_dim: int = 18  # Compatible with IsaacLab reach task
    
    # Observation components
    include_joint_positions: bool = True
    include_joint_velocities: bool = True  
    include_end_effector_pose: bool = True
    include_target_pose: bool = True
    include_previous_actions: bool = False
    include_time_step: bool = True
    
    # Normalization and noise
    normalize_joint_positions: bool = True
    joint_position_noise_std: float = 0.01
    joint_velocity_noise_std: float = 0.01
    joint_velocity_scale: float = 0.1
    
    # Observation history
    observation_history_length: int = 1  # Number of past observations to include
    

@configclass
class RealWorldRewardsCfg:
    """Configuration for reward computation."""
    
    # Reward components and weights
    position_tracking_weight: float = -1.0    # Negative distance reward
    position_tracking_tolerance: float = 0.02  # Target tolerance in meters
    position_success_bonus: float = 10.0       # Bonus for reaching target
    
    orientation_tracking_weight: float = -0.5   # Orientation error penalty
    orientation_tolerance: float = 0.1          # Orientation tolerance in radians
    
    # Regularization penalties
    joint_velocity_penalty_weight: float = -0.001
    action_rate_penalty_weight: float = -0.0001
    joint_torque_penalty_weight: float = -0.00001
    
    # Safety penalties
    joint_limit_penalty_weight: float = -1.0
    workspace_violation_penalty: float = -5.0
    collision_penalty: float = -10.0
    
    # Reward shaping
    use_shaped_rewards: bool = True
    reward_shaping_potential_scale: float = 1.0


@configclass
class RealWorldTaskCfg:
    """Configuration for task-specific parameters."""
    
    # Task definition
    task_type: str = "reach"  # "reach", "pick_place", "door_opening"
    
    # Target generation
    target_sampling_mode: str = "uniform_random"  # "uniform_random", "curriculum", "fixed"
    target_position_range: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "x": (0.35, 0.65),
        "y": (-0.2, 0.2), 
        "z": (0.15, 0.5)
    })
    
    # Target orientation (if applicable)
    target_orientation_mode: str = "fixed"  # "fixed", "random", "task_specific"
    fixed_target_orientation: List[float] = [1.0, 0.0, 0.0, 0.0]  # [qw, qx, qy, qz]
    
    # Episode parameters
    episode_length: float = 15.0  # seconds
    success_tolerance: float = 0.02  # meters
    success_bonus: float = 100.0
    
    # Curriculum learning
    curriculum_enabled: bool = False
    curriculum_levels: List[Dict] = field(default_factory=list)


@configclass
class RealWorldSafetyMonitorCfg:
    """Configuration for safety monitoring and emergency handling."""
    
    # Emergency stop conditions
    max_joint_velocity: float = 2.0  # rad/s
    max_end_effector_velocity: float = 1.0  # m/s  
    max_joint_acceleration: float = 10.0  # rad/s²
    max_contact_force: float = 20.0  # N
    
    # Workspace monitoring
    enforce_workspace_limits: bool = True
    workspace_violation_action: str = "stop"  # "stop", "slow", "warn"
    
    # Joint limit monitoring  
    joint_limit_margin: float = 0.1  # radians
    joint_limit_action: str = "stop"  # "stop", "slow", "warn"
    
    # Communication monitoring
    communication_timeout: float = 1.0  # seconds
    max_consecutive_failures: int = 3
    
    # Recovery behaviors
    enable_automatic_recovery: bool = True
    recovery_home_position: List[float] = [0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398]


@configclass
class RealWorldLoggingCfg:
    """Configuration for data logging and monitoring."""
    
    # Logging parameters
    enable_logging: bool = True
    log_frequency: float = 30.0  # Hz
    log_directory: str = "./real_world_logs"
    
    # Data to log
    log_robot_state: bool = True
    log_actions: bool = True  
    log_observations: bool = True
    log_rewards: bool = True
    log_safety_metrics: bool = True
    
    # Performance monitoring
    monitor_performance: bool = True
    performance_metrics: List[str] = field(default_factory=lambda: [
        "success_rate", "average_episode_length", "average_reward",
        "safety_violations", "communication_errors"
    ])
    
    # Video recording (if cameras available)
    record_video: bool = False
    video_fps: float = 10.0
    video_resolution: Tuple[int, int] = (640, 480)


@configclass
class RealWorldEnvCfg(ManagerBasedRLEnvCfg if ISAACLAB_AVAILABLE else object):
    """
    Main configuration class for real-world Franka robot environment.
    
    This class provides IsaacLab-compatible configuration while adding
    real-world specific parameters for robot control and safety.
    """
    
    # Robot and hardware configuration
    robot: RealWorldRobotCfg = RealWorldRobotCfg()
    
    # Environment components
    actions: RealWorldActionsCfg = RealWorldActionsCfg()
    observations: RealWorldObservationsCfg = RealWorldObservationsCfg()
    rewards: RealWorldRewardsCfg = RealWorldRewardsCfg()
    task: RealWorldTaskCfg = RealWorldTaskCfg()
    
    # Safety and monitoring
    safety: RealWorldSafetyMonitorCfg = RealWorldSafetyMonitorCfg()
    logging: RealWorldLoggingCfg = RealWorldLoggingCfg()
    
    # Environment settings
    num_envs: int = 1  # Real robot = single environment
    episode_length_s: float = 15.0
    decimation: int = 1  # No decimation for real robot (full control frequency)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure single environment for real robot
        self.num_envs = 1
        
        # Synchronize episode length with task configuration
        self.episode_length_s = self.task.episode_length
        
        # Validate configuration
        self._validate_config()
        
        # Set device to CPU for real robot (no GPU simulation)
        self.sim = None  # No simulation for real robot
        
        if ISAACLAB_AVAILABLE:
            # Call parent post_init if available
            try:
                super().__post_init__()
            except:
                pass  # Skip if parent doesn't have __post_init__
        
    def _validate_config(self):
        """Validate configuration parameters for consistency and safety."""
        
        # Check workspace limits are reasonable
        x_min, x_max, y_min, y_max, z_min, z_max = self.robot.workspace_limits
        assert x_max > x_min and y_max > y_min and z_max > z_min, \
            "Invalid workspace limits"
        assert z_min > 0, "Workspace must be above ground (z > 0)"
        
        # Check target range is within workspace
        task_x_min, task_x_max = self.task.target_position_range["x"]
        task_y_min, task_y_max = self.task.target_position_range["y"]
        task_z_min, task_z_max = self.task.target_position_range["z"]
        
        assert (task_x_min >= x_min and task_x_max <= x_max and
                task_y_min >= y_min and task_y_max <= y_max and  
                task_z_min >= z_min and task_z_max <= z_max), \
            "Target range exceeds workspace limits"
            
        # Check safety parameters
        assert self.robot.relative_dynamics_factor <= 1.0, \
            "Dynamics factor should be <= 1.0 for safety"
        assert self.robot.joint_limits_buffer > 0, \
            "Joint limits buffer should be positive"
            
        # Check control frequency is reasonable
        assert 1.0 <= self.robot.control_frequency <= 100.0, \
            "Control frequency should be between 1-100 Hz"


# Predefined configurations for common use cases

@configclass
class RealWorldFrankaReachCfg(RealWorldEnvCfg):
    """Configuration for Franka reaching task (basic setup)."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Task-specific settings for reaching
        self.task.task_type = "reach"
        self.task.episode_length = 15.0
        self.task.success_tolerance = 0.02
        
        # Conservative control for safety
        self.robot.control_frequency = 20.0  # Lower frequency for stability
        self.robot.relative_dynamics_factor = 0.15  # Very conservative
        
        # Action configuration
        self.actions.action_scale = 0.05  # Small actions for precision
        self.actions.action_filter_enabled = True
        

@configclass  
class RealWorldFrankaReachCfg_Safe(RealWorldFrankaReachCfg):
    """Ultra-safe configuration for initial testing."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Maximum safety settings
        self.robot.relative_dynamics_factor = 0.1
        self.robot.joint_limits_buffer = 0.1
        self.actions.action_scale = 0.02
        
        # Restricted workspace
        self.robot.workspace_limits = [0.4, 0.7, -0.2, 0.2, 0.2, 0.5]
        self.task.target_position_range = {
            "x": (0.45, 0.65),
            "y": (-0.15, 0.15),
            "z": (0.25, 0.45)
        }
        
        # Enhanced safety monitoring
        self.safety.max_joint_velocity = 1.0  # Very conservative
        self.safety.max_end_effector_velocity = 0.5
        

@configclass
class RealWorldFrankaReachCfg_Performance(RealWorldFrankaReachCfg):
    """Higher performance configuration for trained policies."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Faster control for trained policies
        self.robot.control_frequency = 30.0
        self.robot.relative_dynamics_factor = 0.3
        
        # Larger action scale
        self.actions.action_scale = 0.1
        
        # Full workspace utilization
        self.task.target_position_range = {
            "x": (0.3, 0.7),
            "y": (-0.3, 0.3),
            "z": (0.1, 0.6)
        }


# Factory functions for easy configuration creation

def get_safe_config() -> RealWorldFrankaReachCfg_Safe:
    """Get ultra-safe configuration for initial testing."""
    return RealWorldFrankaReachCfg_Safe()


def get_standard_config() -> RealWorldFrankaReachCfg:
    """Get standard configuration for normal operation."""
    return RealWorldFrankaReachCfg()


def get_performance_config() -> RealWorldFrankaReachCfg_Performance:
    """Get performance configuration for trained policies."""
    return RealWorldFrankaReachCfg_Performance()


def create_custom_config(**kwargs) -> RealWorldEnvCfg:
    """Create custom configuration with specified parameters.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        Custom configuration instance
    """
    cfg = RealWorldFrankaReachCfg()
    
    # Override specified parameters
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            # Handle nested attribute setting (e.g., "robot.control_frequency")
            parts = key.split('.')
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
    
    return cfg


# Example usage and validation
if __name__ == "__main__":
    print("Testing real-world environment configurations...")
    
    # Test different configurations
    configs = [
        ("Safe", get_safe_config()),
        ("Standard", get_standard_config()), 
        ("Performance", get_performance_config())
    ]
    
    for name, cfg in configs:
        print(f"\n{name} Configuration:")
        print(f"  Control frequency: {cfg.robot.control_frequency} Hz")
        print(f"  Dynamics factor: {cfg.robot.relative_dynamics_factor}")
        print(f"  Action scale: {cfg.actions.action_scale}")
        print(f"  Episode length: {cfg.episode_length_s} s")
        print(f"  Workspace: {cfg.robot.workspace_limits}")
        
        # Validate configuration
        try:
            cfg._validate_config()
            print(f"  ✓ Configuration valid")
        except Exception as e:
            print(f"  ✗ Configuration error: {e}")
    
    # Test custom configuration
    print(f"\nCustom Configuration:")
    custom_cfg = create_custom_config(
        episode_length_s=20.0,
        robot_control_frequency=25.0,
        actions_action_scale=0.08
    )
    print(f"  Episode length: {custom_cfg.episode_length_s} s")
    print(f"  Control frequency: {custom_cfg.robot.control_frequency} Hz") 
    print(f"  Action scale: {custom_cfg.actions.action_scale}")
    
    print("\n✓ Configuration system test completed")