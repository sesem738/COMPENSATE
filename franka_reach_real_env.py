import gymnasium as gym
import time
import numpy as np

from franky import Robot, Gripper, JointMotion


class ReachingFranka(gym.Env):
    def __init__(self, robot_ip="172.16.0.2", device="cuda:0"):
        super().__init__()
        
        self.device = device
        self.robot_ip = robot_ip

        # spaces - only joint control
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(18,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # init real franka with franky
        print(f"Connecting to robot at {robot_ip}...")
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip)
        print("Robot connected")

        # Set robot to moderate speed for safety
        self.robot.relative_dynamics_factor = 0.1  # 10% of max speed

        self.dt = 1 / 30.0
        self.action_scale = 2.5
        self.dof_vel_scale = 0.1
        self.max_episode_length = 1000
        self.robot_dof_speed_scales = 1
        self.target_pos = np.array([0.65, 0.2, 0.2])
        self.robot_default_dof_pos = np.array([0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398])  # radians
        
        # Franka joint limits (standard values)
        self.robot_dof_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.robot_dof_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        self.progress_buf = 1
        self.obs_buf = np.zeros((18,), dtype=np.float32)

    def _get_observation_reward_done(self):
        # get robot state using franky
        cartesian_state = self.robot.current_cartesian_state
        joint_state = self.robot.current_joint_state
        
        robot_dof_pos = np.array(joint_state.position)  # joint positions
        robot_dof_vel = np.array(joint_state.velocity)  # joint velocities
        
        # Get end effector position from cartesian state
        end_effector_pose = cartesian_state.pose.end_effector_pose
        end_effector_pos = np.array([
            end_effector_pose.translation[0],
            end_effector_pose.translation[1], 
            end_effector_pose.translation[2]
        ])

        dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) / (
                self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_dof_vel * self.dof_vel_scale

        self.obs_buf[0] = self.progress_buf / float(self.max_episode_length)
        self.obs_buf[1:8] = dof_pos_scaled
        self.obs_buf[8:15] = dof_vel_scaled
        self.obs_buf[15:18] = self.target_pos

        # reward
        distance = np.linalg.norm(end_effector_pos - self.target_pos)
        reward = -distance

        # done
        done = bool(self.progress_buf >= self.max_episode_length - 1)
        done = done or bool(distance <= 0.04)

        print("Distance:", distance)
        if done:
            print("Target or Maximum episode length reached")
            time.sleep(1)

        return self.obs_buf, reward, done

    def reset(self):
        print("Resetting...")

        # Reset robot to home position using franky
        home_motion = JointMotion(self.robot_default_dof_pos)
        self.robot.move(home_motion)
        self.gripper.move(width=0.0, speed=0.1)
        
        # Move to random position near default
        dof_pos = self.robot_default_dof_pos + 0.25 * (np.random.rand(7) - 0.5)
        random_motion = JointMotion(dof_pos)
        self.robot.move(random_motion)

        # get target position from prompt
        while True:
            try:
                print("Enter target position (X, Y, Z) in meters")
                raw = input("or press [Enter] key for a random target position: ")
                if raw:
                    self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                else:
                    noise = (2 * np.random.rand(3) - 1) * np.array([0.25, 0.25, 0.10])
                    self.target_pos = np.array([0.5, 0.0, 0.3]) + noise
                    if self.target_pos[2] < 0.1:
                        self.target_pos[2] = 0.1
                print("Target position:", self.target_pos)
                break
            except ValueError:
                print("Invalid input. Try something like: 0.65, 0.0, 0.2")

        input("Press [Enter] to continue")

        self.progress_buf = 0
        observation, _, _ = self._get_observation_reward_done()
        return observation, {}

    def step(self, action):
        self.progress_buf += 1

        # Joint space control only
        current_state = self.robot.current_joint_state
        current_joints = np.array(current_state.position)
        target_joints = current_joints + (self.robot_dof_speed_scales * self.dt * action * self.action_scale)
        
        print("End-Effect Position:", self.robot.current_cartesian_state.pose.end_effector_pose.translation)


        # Use franky joint motion
        motion = JointMotion(target_joints)
        
        # For safety, set lower dynamics factor during stepping
        original_factor = self.robot.relative_dynamics_factor
        self.robot.relative_dynamics_factor = 0.1  # 5% speed during fine control
        
        try:
            self.robot.move(motion, asynchronous=True)
        except Exception as e:
            print(f"Motion failed: {e}")
            # Reset to safer dynamics factor
            self.robot.relative_dynamics_factor = original_factor
        else:
            self.robot.relative_dynamics_factor = original_factor

        # Small delay for control stability
        time.sleep(1/30)

        observation, reward, done = self._get_observation_reward_done()
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, {}

    def render(self, *_args, **_kwargs):
        pass

    def close(self):
        # No explicit cleanup needed for franky
        pass


