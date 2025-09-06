#!/usr/bin/env python3

"""
Simple examples for the real-world Franka environment.
Demonstrates basic usage following the style of franka_reach_real_env.py.
"""

import numpy as np
import torch
import time
from franka_real_simple import make_franka_real_env


def example_basic_usage():
    """Basic environment usage - similar to original franka_reach_real_env.py"""
    print("=== Basic Usage Example ===")
    
    try:
        # Create environment with safe settings
        env = make_franka_real_env(
            robot_ip="172.16.0.2",  # Update to your robot's IP
            config="safe",          # Start with safe configuration
            auto_target=True        # Generate targets automatically
        )
        
        print("Environment created successfully")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Reset environment
        obs, info = env.reset()
        print(f"Reset complete - Target: {info['target_position']}")
        
        # Run simple episode
        for step in range(20):
            # Small random action (very conservative)
            action = np.random.randn(7) * 0.03  # Small actions for safety
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step:2d}: Reward={reward:6.3f}, Distance={info['distance_to_target']:.4f}")
            
            if terminated or truncated:
                success = info.get('is_success', False)
                print(f"Episode finished - Success: {success}")
                break
        
        env.close()
        print("✓ Basic usage example completed\n")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if no robot is connected\n")


def example_different_configs():
    """Test different configuration presets"""
    print("=== Configuration Examples ===")
    
    configs = ["safe", "standard", "performance", "test"]
    
    for config_name in configs:
        print(f"\n{config_name.title()} Configuration:")
        
        try:
            env = make_franka_real_env(
                robot_ip="172.16.0.2",
                config=config_name,
                auto_target=True
            )
            
            print(f"  Action scale: {env.action_scale}")
            print(f"  Max episode length: {env.max_episode_length}")
            print(f"  Control frequency: {env.control_frequency} Hz")
            print(f"  Success distance: {env.success_distance} m")
            print(f"  Workspace X: {env.workspace_limits['x']}")
            
            env.close()
            
        except Exception as e:
            print(f"  Error creating config: {e}")
    
    print("\n✓ Configuration examples completed\n")


def example_isaaclab_compatibility():
    """Test IsaacLab-compatible wrapper"""
    print("=== IsaacLab Compatibility Example ===")
    
    try:
        # Create environment with IsaacLab wrapper
        env = make_franka_real_env(
            robot_ip="172.16.0.2",
            config="test",           # Short episodes for testing
            auto_target=True,
            isaaclab_wrapper=True    # Get tensor I/O
        )
        
        print("IsaacLab wrapper created")
        print(f"Number of environments: {env.num_envs}")
        print(f"Device: {env.device}")
        
        # Reset - returns dictionary with policy key
        obs_dict, info = env.reset()
        print(f"Observation keys: {list(obs_dict.keys())}")
        print(f"Policy observation shape: {obs_dict['policy'].shape}")
        print(f"Policy observation device: {obs_dict['policy'].device}")
        
        # Step with tensor action
        action = torch.randn(1, 7, device=env.device) * 0.02  # Batch dimension required
        
        obs_dict, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step completed:")
        print(f"  Reward shape: {reward.shape}, device: {reward.device}")
        print(f"  Terminated shape: {terminated.shape}, device: {terminated.device}")
        print(f"  Info: {info}")
        
        env.close()
        print("✓ IsaacLab compatibility example completed\n")
        
    except Exception as e:
        print(f"Error: {e}\n")


def example_simple_policy():
    """Example with a simple reaching policy"""
    print("=== Simple Policy Example ===")
    
    class SimpleReachingPolicy:
        """Very simple policy that moves towards neutral position"""
        
        def __init__(self, action_scale=0.05):
            self.action_scale = action_scale
            # Franka neutral position  
            self.neutral_joints = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
            
        def __call__(self, observation):
            # Extract joint positions from observation (indices 1-8)
            current_joints_normalized = observation[1:8]
            
            # Convert back to actual joint positions (rough approximation)
            # In real implementation, you'd want the exact denormalization
            current_joints = current_joints_normalized  # Simplified
            
            # Simple P-controller towards neutral position
            joint_error = self.neutral_joints[:7] - current_joints
            action = joint_error * 0.1  # Proportional gain
            
            # Clip action
            action = np.clip(action, -self.action_scale, self.action_scale)
            
            return action
    
    try:
        env = make_franka_real_env(
            robot_ip="172.16.0.2",
            config="safe",
            auto_target=True
        )
        
        policy = SimpleReachingPolicy(action_scale=0.03)
        
        # Run multiple episodes
        num_episodes = 3
        for episode in range(num_episodes):
            obs, info = env.reset()
            print(f"\nEpisode {episode + 1}, Target: {info['target_position']}")
            
            episode_reward = 0
            for step in range(50):
                # Get action from policy
                action = policy(obs)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if step % 10 == 0:
                    print(f"  Step {step:2d}: Distance={info['distance_to_target']:.4f}")
                
                if terminated or truncated:
                    success = info.get('is_success', False)
                    print(f"  Episode finished: Success={success}, Reward={episode_reward:.2f}")
                    break
        
        env.close()
        print("\n✓ Simple policy example completed\n")
        
    except Exception as e:
        print(f"Error: {e}\n")


def example_gymnasium_integration():
    """Test gymnasium environment registration"""
    print("=== Gymnasium Integration Example ===")
    
    try:
        import gymnasium as gym
        
        # List available environments
        available_envs = [
            "FrankaReal-Safe-v0",
            "FrankaReal-v0", 
            "FrankaReal-Performance-v0"
        ]
        
        print("Available registered environments:")
        for env_id in available_envs:
            print(f"  - {env_id}")
        
        # Create environment using gym.make
        env = gym.make("FrankaReal-Safe-v0")
        print(f"\nCreated environment: {env}")
        
        # Test basic interface
        obs, info = env.reset()
        print(f"Reset successful, observation shape: {obs.shape}")
        
        action = np.random.randn(7) * 0.02
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step successful, reward: {reward:.4f}")
        
        env.close()
        print("✓ Gymnasium integration example completed\n")
        
    except ImportError:
        print("Gymnasium not available - skipping\n")
    except Exception as e:
        print(f"Error: {e}\n")


def example_safety_features():
    """Demonstrate safety features"""
    print("=== Safety Features Example ===")
    
    try:
        env = make_franka_real_env(
            robot_ip="172.16.0.2",
            config="safe",
            auto_target=True
        )
        
        obs, info = env.reset()
        print("Testing safety features...")
        
        # Test 1: Normal safe action
        safe_action = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01])
        obs, reward, terminated, truncated, info = env.step(safe_action)
        print(f"Safe action: Action safe={info.get('action_was_safe', 'Unknown')}")
        
        # Test 2: Large action (will be automatically scaled down)
        large_action = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        print(f"\nTesting large action with norm: {np.linalg.norm(large_action):.3f}")
        obs, reward, terminated, truncated, info = env.step(large_action)
        print(f"Large action: Action safe={info.get('action_was_safe', 'Unknown')}")
        
        # Test 3: Check safety parameters
        print(f"\nSafety parameters:")
        print(f"  Max action norm: {env.max_action_norm}")
        print(f"  Joint buffer: {env.joint_buffer}")
        print(f"  Workspace limits: {env.workspace_limits}")
        print(f"  Success distance: {env.success_distance}")
        
        env.close()
        print("\n✓ Safety features example completed\n")
        
    except Exception as e:
        print(f"Error: {e}\n")


def example_training_template():
    """Template for training integration"""
    print("=== Training Integration Template ===")
    
    training_code = '''
# Example integration with training frameworks

def train_with_real_robot():
    from franka_real_simple import make_franka_real_env
    
    # Create environment
    env = make_franka_real_env(
        robot_ip="YOUR_ROBOT_IP",
        config="standard",      # Adjust based on your needs
        auto_target=True,       # No manual interaction
        isaaclab_wrapper=True   # If you need tensor I/O
    )
    
    try:
        for episode in range(num_training_episodes):
            obs_dict, info = env.reset()
            
            for step in range(max_episode_length):
                # Get action from your policy
                action = your_policy(obs_dict["policy"])
                
                # Step environment
                obs_dict, reward, terminated, truncated, info = env.step(action)
                
                # Store experience for training
                store_experience(obs_dict["policy"], action, reward, ...)
                
                if terminated or truncated:
                    break
            
            # Update policy
            update_policy()
            
            # Log progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Success rate: {calculate_success_rate()}")
    
    finally:
        env.close()

# Framework-specific examples:

# 1. With RSL-RL
from rsl_rl.runners import OnPolicyRunner
env = make_franka_real_env(config="standard", isaaclab_wrapper=True)
runner = OnPolicyRunner(env, cfg, log_dir="./logs")

# 2. With Stable-Baselines3
from stable_baselines3 import PPO
env = make_franka_real_env(config="standard")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 3. With custom training loop
env = make_franka_real_env(config="standard", auto_target=True)
for episode in range(1000):
    obs, info = env.reset()
    # ... your training loop
'''
    
    print(training_code)
    print("✓ Training template provided\n")


def main():
    """Run all examples"""
    print("Simple Real-World Franka Environment Examples")
    print("=" * 50)
    print("Note: These examples expect robot at IP 172.16.0.2")
    print("Update robot_ip in each example to match your setup")
    print("=" * 50)
    
    # Run examples (most will fail without robot, but show usage)
    example_basic_usage()
    example_different_configs()
    example_isaaclab_compatibility()
    example_simple_policy()
    example_gymnasium_integration()
    example_safety_features()
    example_training_template()
    
    print("=" * 50)
    print("All examples completed!")
    print("\nTo run with real robot:")
    print("1. Update robot_ip to your robot's IP address")
    print("2. Ensure robot is powered on and network accessible")  
    print("3. Start with config='safe' for initial testing")
    print("4. Use auto_target=True to avoid manual input")
    print("5. Keep emergency stop button accessible")
    print("=" * 50)


if __name__ == "__main__":
    main()