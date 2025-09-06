import gymnasium as gym
import time
import numpy as np
import torch

from franky import Robot, Gripper, JointMotion


class FrankaReachPose(gym.Env):
    
    def __init__(self, robot_ip="172.16.0.2", device="cuda:0", auto_target=True):
        super().__init__()
        
        self.device = device
        self.robot_ip = robot_ip
        self.auto_target = auto_target
        
        # Gym spaces - compatible with IsaacLab reach task
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(28,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-4, high=4, shape=(7,), dtype=np.float32)
        
        # Connect to robot
        print(f"Connecting to robot at {robot_ip}...")
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip) 
        print("Robot connected")
        
        # Safety settings
        self.robot.relative_dynamics_factor = 0.15  # 15% of max speed for safety
        
        # Control parameters
        self.dt = 1 / 30.0
        self.action_scale = 0.5  # Smaller scale for safety
        self.dof_vel_scale = 0.1
        self.max_episode_length = 500
        
        # Target and home positions
        self.target_pos = np.array([0.5, 0.0, 0.3, 0.123161, 0.986249, -0.000304419, 0.110198])
        self.robot_default_dof_pos = np.array([0, -0.569, 0, -2.81, 0, 3.037, 0.741])
        self.previous_action = np.zeros(7)
        
        # Joint limits for safety
        self.robot_dof_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.robot_dof_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Workspace limits for safety
        self.workspace_limits = {
            'x': (0.3, 0.7),
            'y': (-0.23, 0.23), 
            'z': (0.3, 0.8)
        }
        
        # State tracking
        self.progress_buf = 0
        self.obs_buf = np.zeros((28,), dtype=np.float32)  # Fixed: 28D to match observation_space
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
            end_effector_pose.translation[2],
        ])
        
        
        self.obs_buf[:7] = robot_dof_pos - self.robot_default_dof_pos
        self.obs_buf[7:14] = robot_dof_vel
        self.obs_buf[14:21] = self.target_pos  
        self.obs_buf[21:28] = self.previous_action
        
        
        # Calculate reward (negative distance to target)
        distance = np.linalg.norm(end_effector_pos - self.target_pos[:3])
        reward = -distance
        
        # Add success bonus
        if distance <= 0.03:  # 3cm tolerance
            reward += 10.0
            
        # Check termination conditions
        done = bool(self.progress_buf >= self.max_episode_length - 1)
        done = done or bool(distance <= 0.001)  # Success termination
        
            
        if done:
            success = distance <= 0.03
            print(f"Episode finished - Success: {success}, Distance: {distance:.4f}")
            
        return self.obs_buf, reward, done
        
    def _generate_random_target(self):
        """Generate a random target within workspace."""
        x_min, x_max = self.workspace_limits['x']
        y_min, y_max = self.workspace_limits['y']
        z_min, z_max = self.workspace_limits['z']

        q = np.array([0.123161,0.986249,-0.000304419,0.110198])
        
        self.target_pos = np.array([
            np.random.uniform(x_min + 0.1, x_max - 0.1),
            np.random.uniform(y_min + 0.1, y_max - 0.1),
            np.random.uniform(z_min + 0.1, z_max - 0.1),
            q[3],
            q[0],
            q[1],
            q[2]
        ])
        
        
    def reset(self, seed=None, options=None):
        
        self.episode_count += 1
        print(f"\n=== Episode {self.episode_count} Reset ===")

        # Move to home position
        print("Moving to home position...")
        home_motion = JointMotion(self.robot_default_dof_pos)
        self.robot.move(home_motion)
        self.gripper.move(width=0, speed=0.1)
        
        # Set target position
        if self.auto_target:
            # Fixed: Support auto_target mode for automated evaluation
            self._generate_random_target()
            print(f"Auto-generated target position: {self.target_pos}")
        else:
            # get target position from prompt
            while True:
                try:
                    print("Enter target position (X, Y, Z, Q1, Q2, Q3, Q4) in meters")
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
        
        # Get current joint positions
        current_state = self.robot.current_joint_state
        current_joints = np.array(current_state.position)
        
        # Calculate target joint positions
        target_joints = self.robot_default_dof_pos + (action * self.action_scale)
        
        # Execute motion
        motion = JointMotion(target_joints)
        
        try:
            self.robot.move(motion, asynchronous=True)
        except Exception as e:
            print(f"Motion failed: {e}")
            # Don't crash, just continue with current position
            
        # Control rate limiting
        time.sleep(self.dt)
        
        # Store the action for next observation
        self.previous_action = action.copy()
        
        # Get observation, reward, and done status
        observation, reward, terminated = self._get_observation_reward_done()
        truncated = False  # We use terminated for both success and timeout
        
        end_effector_pos = np.array([
            self.robot.current_cartesian_state.pose.end_effector_pose.translation[0],
            self.robot.current_cartesian_state.pose.end_effector_pose.translation[1],
            self.robot.current_cartesian_state.pose.end_effector_pose.translation[2]
        ])
        
        # Info dictionary
        info = {
            'episode_step': self.progress_buf,
            'is_success': terminated and reward > 0,  # Success if terminated with positive reward
            'distance_to_target': np.linalg.norm(end_effector_pos - self.target_pos[:3])
        }
        
        return observation, reward, terminated, truncated, info
        
    def render(self, mode='human'):
        """Render the environment (no-op for real robot)."""
        pass
        
    def close(self):
        """Clean up environment."""
        print("Closing environment...")
        try:
            # Move to safe position before closing
            home_motion = JointMotion(self.robot_default_dof_pos)
            self.robot.move(home_motion)
            print("Robot returned to home position")
        except:
            print("Warning: Could not return robot to home position")