"""
Simple configuration system for real-world Franka robot environment.
Follows the style of the original franka_reach_real_env.py with basic customization options.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class FrankaRealConfig:
    """Simple configuration for real-world Franka environment."""
    
    # Robot connection
    robot_ip: str = "172.16.0.2"
    device: str = "cuda:0"
    auto_target: bool = True  # Generate targets automatically vs user input
    
    # Control parameters
    control_frequency: float = 30.0  # Hz
    action_scale: float = 0.1  # Scale factor for actions
    dynamics_factor: float = 0.15  # Robot speed (0.0-1.0)
    
    # Episode parameters
    max_episode_length: int = 500
    success_distance: float = 0.03  # meters
    success_reward: float = 10.0
    
    # Safety parameters
    max_action_norm: float = 0.5  # Maximum action magnitude
    joint_buffer: float = 0.1  # Safety buffer from joint limits (radians)
    workspace_buffer: float = 0.1  # Safety buffer for workspace (meters)
    
    # Workspace limits [min, max] for [x, y, z]
    workspace_x: Tuple[float, float] = (0.3, 0.8)
    workspace_y: Tuple[float, float] = (-0.4, 0.4)
    workspace_z: Tuple[float, float] = (0.1, 0.8)
    
    # Default positions
    home_position: np.ndarray = None
    default_target: np.ndarray = None
    
    def __post_init__(self):
        """Set default arrays after initialization."""
        if self.home_position is None:
            self.home_position = np.array([0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398])
        
        if self.default_target is None:
            self.default_target = np.array([0.5, 0.0, 0.3])
    
    @property
    def workspace_limits(self) -> Dict[str, Tuple[float, float]]:
        """Get workspace limits as dictionary."""
        return {
            'x': self.workspace_x,
            'y': self.workspace_y,
            'z': self.workspace_z
        }
    
    @property 
    def dt(self) -> float:
        """Control timestep."""
        return 1.0 / self.control_frequency


# Predefined configurations

class SafeConfig(FrankaRealConfig):
    """Ultra-safe configuration for initial testing."""
    
    def __init__(self):
        super().__init__()
        self.dynamics_factor = 0.1  # Very slow
        self.action_scale = 0.05  # Small actions
        self.max_action_norm = 0.2  # Conservative action limit
        self.joint_buffer = 0.15  # Large safety buffer
        self.workspace_x = (0.4, 0.7)  # Restricted workspace
        self.workspace_y = (-0.2, 0.2)
        self.workspace_z = (0.2, 0.6)


class StandardConfig(FrankaRealConfig):
    """Standard configuration for normal operation."""
    
    def __init__(self):
        super().__init__()
        # Uses default values which are already reasonable


class PerformanceConfig(FrankaRealConfig):
    """Higher performance configuration for trained policies."""
    
    def __init__(self):
        super().__init__()
        self.dynamics_factor = 0.25  # Faster movement
        self.action_scale = 0.15  # Larger actions
        self.control_frequency = 40.0  # Higher frequency
        self.max_episode_length = 300  # Shorter episodes
        self.workspace_x = (0.25, 0.85)  # Larger workspace
        self.workspace_y = (-0.45, 0.45)
        self.workspace_z = (0.05, 0.85)


class TestConfig(FrankaRealConfig):
    """Configuration for testing without user interaction."""
    
    def __init__(self):
        super().__init__()
        self.auto_target = True  # Always auto-generate targets
        self.max_episode_length = 100  # Short episodes
        self.dynamics_factor = 0.1  # Very safe
        self.action_scale = 0.03  # Very small actions


# Factory functions

def get_config(config_name: str = "standard") -> FrankaRealConfig:
    """Get predefined configuration by name.
    
    Args:
        config_name: One of 'safe', 'standard', 'performance', 'test'
        
    Returns:
        Configuration instance
    """
    configs = {
        'safe': SafeConfig,
        'standard': StandardConfig,
        'performance': PerformanceConfig, 
        'test': TestConfig
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown config '{config_name}', using 'standard'")
        config_name = 'standard'
        
    return configs[config_name]()


def create_custom_config(**kwargs) -> FrankaRealConfig:
    """Create custom configuration with parameter overrides.
    
    Args:
        **kwargs: Parameters to override from default configuration
        
    Returns:
        Custom configuration instance
    """
    config = StandardConfig()
    
    # Override provided parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown parameter '{key}' ignored")
            
    return config


# Environment factory with configuration

def make_franka_env_with_config(config_name: str = "standard", **kwargs):
    """Create Franka environment with specified configuration.
    
    Args:
        config_name: Predefined configuration name
        **kwargs: Additional parameter overrides
        
    Returns:
        Configured environment
    """
    from franka_reach_real_simple import make_franka_real_env
    
    # Get base configuration
    config = get_config(config_name)
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create environment with configuration
    env = make_franka_real_env(
        robot_ip=config.robot_ip,
        device=config.device,
        auto_target=config.auto_target,
        isaaclab_compatible=kwargs.get('isaaclab_compatible', False)
    )
    
    # Apply configuration to environment
    env.robot.relative_dynamics_factor = config.dynamics_factor
    env.action_scale = config.action_scale
    env.max_episode_length = config.max_episode_length
    env.dt = config.dt
    env.workspace_limits = config.workspace_limits
    
    # Override some internal parameters
    env._success_distance = config.success_distance
    env._success_reward = config.success_reward
    env._max_action_norm = config.max_action_norm
    env._joint_buffer = config.joint_buffer
    
    return env


if __name__ == "__main__":
    print("Testing Franka real-world configuration system...")
    
    # Test different configurations
    configs = ['safe', 'standard', 'performance', 'test']
    
    for config_name in configs:
        print(f"\n{config_name.title()} Configuration:")
        config = get_config(config_name)
        
        print(f"  Dynamics factor: {config.dynamics_factor}")
        print(f"  Action scale: {config.action_scale}")
        print(f"  Control freq: {config.control_frequency} Hz")
        print(f"  Episode length: {config.max_episode_length}")
        print(f"  Workspace X: {config.workspace_x}")
        print(f"  Auto target: {config.auto_target}")
    
    # Test custom configuration
    print(f"\nCustom Configuration:")
    custom_config = create_custom_config(
        robot_ip="192.168.1.100",
        action_scale=0.08,
        max_episode_length=200,
        workspace_x=(0.35, 0.75)
    )
    
    print(f"  Robot IP: {custom_config.robot_ip}")
    print(f"  Action scale: {custom_config.action_scale}")
    print(f"  Episode length: {custom_config.max_episode_length}")
    print(f"  Workspace X: {custom_config.workspace_x}")
    
    print("\n✓ Configuration system test completed")