import gym
import numpy as np
import rospy
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from packaging import version

class ReachingFrankaPy(gym.Env):
    """Gym environment for reaching tasks using FrankaPy joint position control."""
    
    def __init__(self, robot_ip="172.16.0.2", device="cuda:0"):
        # Gym API version check
        self._deprecated_api = version.parse(gym.__version__) < version.parse("0.25.0")
        
        # Store configs
        self.device = device
        self.dt = 0.02  # Control at 50Hz
        self.action_scale = 2.5
        self.dof_vel_scale = 0.1
        self.max_episode_length = 100
        
        # Initialize robot
        print("Connecting to robot...")
        self.franka_arm = FrankaArm()
        print("Robot connected")
        
        # Joint limits from FrankaPy
        self.robot_dof_lower_limits = np.array([
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
        ])
        self.robot_dof_upper_limits = np.array([
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
        ])
        
        # Default home position
        self.robot_default_dof_pos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785])
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-1000, high=1000, shape=(18,), dtype=np.float32
        )
        
        # Initialize ROS publisher for dynamic control
        self.pub = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )
        self.rate = rospy.Rate(1/self.dt)
        self.msg_id = 0
        
        # Initialize target and state tracking
        self.target_pos = np.array([0.5, 0.0, 0.5])  # Default target in workspace
        self.progress_buf = 0
        self.obs_buf = np.zeros((18,), dtype=np.float32)

    def _get_observation_reward_done(self):
        """Get current observation, compute reward and done condition."""
        # Get robot state
        robot_dof_pos = np.array(self.franka_arm.get_joints())
        robot_dof_vel = np.zeros(7)  # FrankaPy doesn't provide easy velocity access
        end_effector_pos = self.franka_arm.get_pose().translation
        
        # Scale joint positions to [-1, 1]
        dof_pos_scaled = (
            2.0 * (robot_dof_pos - self.robot_dof_lower_limits) /
            (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        )
        
        # Scale velocities
        dof_vel_scaled = robot_dof_vel * self.dof_vel_scale
        
        # Update observation buffer
        self.obs_buf[0] = self.progress_buf / float(self.max_episode_length)
        self.obs_buf[1:8] = dof_pos_scaled
        self.obs_buf[8:15] = dof_vel_scaled
        self.obs_buf[15:18] = self.target_pos
        
        # Compute reward based on distance to target
        distance = np.linalg.norm(end_effector_pos - self.target_pos)
        reward = -distance
        
        # Check if done
        done = self.progress_buf >= self.max_episode_length - 1
        done = done or distance <= 0.075  # Success threshold
        
        print(f"Distance: {distance:.3f}")
        if done:
            print("Target reached or maximum episode length reached")
            rospy.sleep(1.0)
            
        return self.obs_buf, reward, done

    def reset(self):
        """Reset the environment."""
        print("Resetting environment...")
        
        # Stop any ongoing motion
        self.franka_arm.stop_skill()
        
        # Go to home position
        self.franka_arm.reset_joints()
        rospy.sleep(0.5)
        
        # Add some random noise to initial position
        dof_pos = self.robot_default_dof_pos + 0.25 * (np.random.rand(7) - 0.5)
        self.franka_arm.goto_joints(dof_pos, duration=3.0)
        rospy.sleep(0.5)
        
        # Set random target or get from user input
        while True:
            try:
                print("Enter target position (X, Y, Z) in meters")
                raw = input("or press [Enter] key for a random target position: ")
                if raw:
                    self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                else:
                    noise = (2 * np.random.rand(3) - 1) * np.array([0.25, 0.25, 0.10])
                    self.target_pos = np.array([0.5, 0.0, 0.2]) + noise
                print("Target position:", self.target_pos)
                break
            except ValueError:
                print("Invalid input. Try something like: 0.65, 0.0, 0.2")
        
        input("Press [Enter] to start episode")

        self.init_time = rospy.Time.now().to_time()
        
        # Reset episode tracking
        self.progress_buf = 0
        self.msg_id = 0
        
        # Initialize dynamic control
        self.franka_arm.goto_joints(
            self.franka_arm.get_joints(),
            duration=self.max_episode_length * self.dt,
            dynamic=True,
            buffer_time=self.max_episode_length * self.dt * 2
        )
        
        # Get initial observation
        observation, reward, done = self._get_observation_reward_done()
        
        if self._deprecated_api:
            return observation
        else:
            return observation, {}

    def step(self, action):
        """Execute one environment step."""
        self.progress_buf += 1
        
        # Scale action and compute target joint positions
        current_joints = np.array(self.franka_arm.get_joints())
        target_joints = current_joints + (action * self.action_scale * self.dt)
        
        # Clip to joint limits
        target_joints = np.clip(
            target_joints, 
            self.robot_dof_lower_limits, 
            self.robot_dof_upper_limits
        )
        
        # Create and publish message
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=self.msg_id, 
            timestamp=rospy.Time.now().to_time() - self.init_time,
            joints=target_joints.tolist()
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, 
                SensorDataMessageType.JOINT_POSITION
            )
        )
        self.pub.publish(ros_msg)
        self.msg_id += 1
        
        # Control rate
        self.rate.sleep()
        
        # Get observation and compute reward
        observation, reward, done = self._get_observation_reward_done()
        
        if self._deprecated_api:
            return observation, reward, done, {}
        else:
            return observation, reward, done, done, {}

    def render(self, *args, **kwargs):
        """Rendering is not implemented for real robot."""
        pass

    def close(self):
        """Clean up resources."""
        self.franka_arm.stop_skill()