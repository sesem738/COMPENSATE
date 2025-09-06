"""
Simple integrated real-world Franka environment.
Combines all components in a single file for easy use, following the style of franka_reach_real_env.py.
"""

import gymnasium as gym
import time
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union

from franky import Robot, Gripper, JointMotion


class FrankaRealEnv(gym.Env):
    """Simple real-world Franka reaching environment with IsaacLab compatibility."""
    
    def __init__(self, 
                 robot_ip: str = "172.16.0.2",
                 device: str = "cuda:0", 
                 config: str = "standard",
                 auto_target: bool = True):
        """Initialize real-world Franka environment.
        
        Args:
            robot_ip: IP address of the Franka robot
            device: Device for tensor operations
            config: Configuration preset ('safe', 'standard', 'performance', 'test')
            auto_target: If True, generates random targets automatically
        """
        super().__init__()
        
        self.robot_ip = robot_ip
        self.device = device
        self.auto_target = auto_target
        
        # Load configuration
        self._load_config(config)
        
        # Gym spaces
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(18,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        # Connect to robot
        print(f"Connecting to robot at {robot_ip}...")
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip)
        print("Robot connected")
        
        # Apply configuration
        self.robot.relative_dynamics_factor = self.dynamics_factor
        
        # Standard Franka parameters
        self.robot_default_dof_pos = np.array([0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398])
        self.robot_dof_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.robot_dof_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # State tracking
        self.progress_buf = 0
        self.obs_buf = np.zeros((18,), dtype=np.float32)
        self.episode_count = 0
        self.target_pos = np.array([0.5, 0.0, 0.3])
        
        # Simple safety checker
        self._init_safety()
        
    def _load_config(self, config: str):
        """Load configuration parameters."""
        configs = {
            'safe': {
                'dynamics_factor': 0.1,
                'action_scale': 0.05,
                'control_frequency': 20.0,
                'max_episode_length': 300,
                'workspace': {'x': (0.4, 0.7), 'y': (-0.2, 0.2), 'z': (0.2, 0.6)},
                'success_distance': 0.04,
                'max_action_norm': 0.2
            },
            'standard': {
                'dynamics_factor': 0.15,
                'action_scale': 0.1,
                'control_frequency': 30.0,
                'max_episode_length': 500,
                'workspace': {'x': (0.3, 0.8), 'y': (-0.4, 0.4), 'z': (0.1, 0.8)},
                'success_distance': 0.03,
                'max_action_norm': 0.5
            },
            'performance': {
                'dynamics_factor': 0.25,
                'action_scale': 0.15,
                'control_frequency': 40.0,
                'max_episode_length': 300,
                'workspace': {'x': (0.25, 0.85), 'y': (-0.45, 0.45), 'z': (0.05, 0.85)},
                'success_distance': 0.02,
                'max_action_norm': 0.7
            },
            'test': {
                'dynamics_factor': 0.1,
                'action_scale': 0.03,
                'control_frequency': 20.0,
                'max_episode_length': 100,
                'workspace': {'x': (0.4, 0.7), 'y': (-0.2, 0.2), 'z': (0.2, 0.6)},
                'success_distance': 0.04,
                'max_action_norm': 0.1
            }
        }
        
        if config not in configs:
            print(f"Warning: Unknown config '{config}', using 'standard'")
            config = 'standard'
            
        cfg = configs[config]
        self.dynamics_factor = cfg['dynamics_factor']
        self.action_scale = cfg['action_scale']
        self.control_frequency = cfg['control_frequency']
        self.max_episode_length = cfg['max_episode_length']
        self.workspace_limits = cfg['workspace']
        self.success_distance = cfg['success_distance']
        self.max_action_norm = cfg['max_action_norm']
        
        self.dt = 1.0 / self.control_frequency
        self.dof_vel_scale = 0.1
        
    def _init_safety(self):
        """Initialize basic safety parameters."""
        self.joint_buffer = 0.1  # Safety buffer from joint limits
        
    def _check_safety(self, action: np.ndarray, current_joints: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Basic safety check and action modification.
        
        Returns:
            Tuple of (safe_action, is_safe)
        """
        safe_action = action.copy()
        warnings = []
        
        # Limit action magnitude
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_action_norm:
            scale_factor = self.max_action_norm / action_norm
            safe_action = action * scale_factor
            warnings.append(f"Action scaled down from {action_norm:.3f} to {self.max_action_norm}")
        
        # Check predicted joint positions
        predicted_joints = current_joints + (self.dt * safe_action * self.action_scale)
        
        # Apply joint limits with safety buffer
        safe_lower = self.robot_dof_lower_limits + self.joint_buffer
        safe_upper = self.robot_dof_upper_limits - self.joint_buffer
        
        violations = []
        for i, (pred, lower, upper) in enumerate(zip(predicted_joints, safe_lower, safe_upper)):
            if pred < lower or pred > upper:
                violations.append(f"Joint {i+1}")
        
        if violations:
            safe_action = safe_action * 0.5  # Reduce action scale
            warnings.append(f"Joint limit risk for: {', '.join(violations)}")
        
        # Print warnings if any
        if warnings:
            print(f"Safety: {'; '.join(warnings)}")
            
        return safe_action, len(violations) == 0
        
    def _generate_random_target(self):
        """Generate random target within workspace."""
        x_min, x_max = self.workspace_limits['x']
        y_min, y_max = self.workspace_limits['y'] 
        z_min, z_max = self.workspace_limits['z']
        
        self.target_pos = np.array([
            np.random.uniform(x_min + 0.05, x_max - 0.05),
            np.random.uniform(y_min + 0.05, y_max - 0.05),
            np.random.uniform(z_min + 0.05, z_max - 0.05)
        ])
        
    def _get_observation_reward_done(self):
        """Get current observation, reward, and termination status."""
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
        
        # Normalize joint positions
        dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) / (
            self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_dof_vel * self.dof_vel_scale
        
        # Build observation (IsaacLab format)
        self.obs_buf[0] = self.progress_buf / float(self.max_episode_length)
        self.obs_buf[1:8] = dof_pos_scaled
        self.obs_buf[8:15] = dof_vel_scaled
        self.obs_buf[15:18] = self.target_pos
        
        # Calculate reward
        distance = np.linalg.norm(end_effector_pos - self.target_pos)
        reward = -distance  # Negative distance
        
        # Success bonus
        success = distance <= self.success_distance
        if success:
            reward += 10.0
            
        # Workspace safety check
        x, y, z = end_effector_pos
        x_min, x_max = self.workspace_limits['x']
        y_min, y_max = self.workspace_limits['y']
        z_min, z_max = self.workspace_limits['z']
        
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            reward -= 2.0  # Penalty for leaving workspace
            print(f"Warning: End effector at ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace")
        
        # Termination conditions
        done = bool(self.progress_buf >= self.max_episode_length - 1)
        done = done or success
        
        # Progress logging
        if self.progress_buf % 20 == 0 or done:
            print(f"Step {self.progress_buf}: Distance={distance:.4f}, Reward={reward:.4f}, Success={success}")
            
        return self.obs_buf.copy(), reward, done
        
    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        if seed is not None:
            np.random.seed(seed)
            
        self.episode_count += 1
        print(f"\n=== Episode {self.episode_count} Reset ===")
        
        # Move to home position
        print("Moving to home position...")
        try:
            home_motion = JointMotion(self.robot_default_dof_pos)
            self.robot.move(home_motion)
            self.gripper.move(width=0.08, speed=0.1)
        except Exception as e:
            print(f"Warning: Reset motion failed: {e}")
        
        # Random start position
        start_offset = 0.1 * (np.random.rand(7) - 0.5)
        start_pos = np.clip(
            self.robot_default_dof_pos + start_offset,
            self.robot_dof_lower_limits + 0.1,
            self.robot_dof_upper_limits - 0.1
        )
        
        try:
            start_motion = JointMotion(start_pos)
            self.robot.move(start_motion)
        except Exception as e:
            print(f"Warning: Start position motion failed: {e}")
        
        # Set target
        if self.auto_target:
            self._generate_random_target()
            print(f"Target: {self.target_pos}")
        else:
            # Interactive target input
            while True:
                try:
                    print("Enter target position (X, Y, Z) in meters")
                    raw = input("or press [Enter] for random: ")
                    if raw:
                        self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                    else:
                        self._generate_random_target()
                    print(f"Target: {self.target_pos}")
                    break
                except ValueError:
                    print("Invalid input. Try: 0.5, 0.0, 0.3")
            input("Press [Enter] to start")
        
        self.progress_buf = 0
        observation, _, _ = self._get_observation_reward_done()
        
        info = {
            'episode': self.episode_count,
            'target_position': self.target_pos.tolist()
        }
        
        return observation, info
        
    def step(self, action):
        """Execute one environment step."""
        self.progress_buf += 1
        
        # Convert action to numpy
        if hasattr(action, 'cpu'):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze()
            
        # Get current joint positions
        try:
            current_state = self.robot.current_joint_state
            current_joints = np.array(current_state.position)
        except Exception as e:
            print(f"Error reading robot state: {e}")
            # Return safe termination
            obs = self.obs_buf.copy()
            return obs, -10.0, True, False, {'error': str(e)}
        
        # Safety check
        safe_action, is_safe = self._check_safety(action, current_joints)
        
        # Calculate target joints
        target_joints = current_joints + (self.dt * safe_action * self.action_scale)
        
        # Execute motion
        try:
            motion = JointMotion(target_joints)
            self.robot.move(motion, asynchronous=True)
        except Exception as e:
            print(f"Motion failed: {e}")
            # Continue without crashing
            
        # Control timing
        time.sleep(self.dt)
        
        # Get observation, reward, done
        observation, reward, terminated = self._get_observation_reward_done()
        truncated = False
        
        # Info
        distance = np.linalg.norm(
            np.array(self.robot.current_cartesian_state.pose.end_effector_pose.translation) - 
            self.target_pos
        ) if hasattr(self.robot, 'current_cartesian_state') else 0.0
        
        info = {
            'episode_step': self.progress_buf,
            'is_success': terminated and reward > 0,
            'distance_to_target': distance,
            'action_was_safe': is_safe
        }
        
        return observation, reward, terminated, truncated, info
        
    def render(self, mode='human'):
        """Render (no-op for real robot)."""
        pass
        
    def close(self):
        """Close environment and return robot to safe position."""
        print("Closing environment...")
        try:
            home_motion = JointMotion(self.robot_default_dof_pos)
            self.robot.move(home_motion)
            print("Robot returned to home position")
        except Exception as e:
            print(f"Warning: Could not return to home: {e}")


class FrankaRealIsaacLabWrapper:
    """IsaacLab-compatible wrapper for tensor operations."""
    
    def __init__(self, env: FrankaRealEnv):
        self.env = env
        self.device = env.device
        self.num_envs = 1
        
    @property
    def observation_space(self):
        return self.env.observation_space
        
    @property 
    def action_space(self):
        return self.env.action_space
        
    def reset(self, env_ids=None, seed=None, options=None):
        """Reset with tensor output."""
        obs, info = self.env.reset(seed=seed, options=options)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return {"policy": obs_tensor}, info
        
    def step(self, actions):
        """Step with tensor I/O."""
        if torch.is_tensor(actions):
            action = actions.cpu().numpy().squeeze()
        else:
            action = np.array(actions).squeeze()
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Convert to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        terminated_tensor = torch.tensor([terminated], dtype=torch.bool, device=self.device)
        truncated_tensor = torch.tensor([truncated], dtype=torch.bool, device=self.device)
        
        return {"policy": obs_tensor}, reward_tensor, terminated_tensor, truncated_tensor, info
        
    def close(self):
        self.env.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Simple factory functions

def make_franka_real_env(robot_ip: str = "172.16.0.2",
                        device: str = "cuda:0",
                        config: str = "standard", 
                        auto_target: bool = True,
                        isaaclab_wrapper: bool = False):
    """Create real-world Franka environment.
    
    Args:
        robot_ip: Robot IP address
        device: Device for tensors
        config: Configuration preset
        auto_target: Automatic target generation
        isaaclab_wrapper: Return IsaacLab-compatible wrapper
        
    Returns:
        Environment instance
    """
    env = FrankaRealEnv(robot_ip, device, config, auto_target)
    
    if isaaclab_wrapper:
        return FrankaRealIsaacLabWrapper(env)
    else:
        return env


# Gymnasium registration
try:
    import gymnasium as gym
    
    gym.register(
        id="FrankaReal-v0",
        entry_point=lambda: make_franka_real_env(config="standard", auto_target=True),
    )
    
    gym.register(
        id="FrankaReal-Safe-v0", 
        entry_point=lambda: make_franka_real_env(config="safe", auto_target=True),
    )
    
    gym.register(
        id="FrankaReal-Performance-v0",
        entry_point=lambda: make_franka_real_env(config="performance", auto_target=True),
    )
    
    print("Real-world Franka environments registered with Gymnasium")
    
except ImportError:
    print("Gymnasium not available - skipping registration")


if __name__ == "__main__":
    print("Testing simple real-world Franka environment...")
    
    try:
        # Create environment
        env = make_franka_real_env(
            robot_ip="172.16.0.2",
            config="safe",  # Use safe config for testing
            auto_target=True,
            isaaclab_wrapper=False
        )
        
        print(f"Environment created: {type(env).__name__}")
        print(f"Config: safe, Auto-target: True")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"Reset successful - observation shape: {obs.shape}")
        print(f"Target: {info['target_position']}")
        
        # Test a few steps
        for step in range(3):
            action = np.random.randn(7) * 0.05  # Very small actions
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step + 1}: Reward={reward:.3f}, Distance={info['distance_to_target']:.3f}")
            
            if terminated:
                print("Episode terminated")
                break
                
        env.close()
        print("✓ Test completed successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if no physical robot is connected")
    
    # Test IsaacLab wrapper
    print("\nTesting IsaacLab wrapper...")
    try:
        env = make_franka_real_env(
            config="test",
            isaaclab_wrapper=True
        )
        
        obs_dict, info = env.reset()
        print(f"IsaacLab reset - obs keys: {obs_dict.keys()}")
        print(f"Policy obs shape: {obs_dict['policy'].shape}")
        
        action = torch.randn(1, 7, device=env.device) * 0.02
        obs_dict, reward, terminated, truncated, info = env.step(action)
        
        print(f"IsaacLab step - reward shape: {reward.shape}, terminated: {terminated}")
        
        env.close()
        print("✓ IsaacLab wrapper test completed")
        
    except Exception as e:
        print(f"IsaacLab wrapper test failed: {e}")
        
    print("\nTo use with real robot:")
    print("1. Update robot_ip to your robot's actual IP")  
    print("2. Start with 'safe' configuration")
    print("3. Use auto_target=True to avoid manual input")
    print("4. Keep emergency stop accessible")