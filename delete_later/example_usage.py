#!/usr/bin/env python3

"""
Example usage of the real-world Franka robot environment.

This script demonstrates various ways to use the real-world environment
interface, from basic testing to integration with training frameworks.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional

# Import the real-world environment components
from real_world_integration import make_real_world_env, register_real_world_envs
from real_world_env_cfg import get_safe_config, get_standard_config, create_custom_config


def example_basic_usage():
    """Basic environment usage example."""
    print("=" * 60)
    print("BASIC ENVIRONMENT USAGE")
    print("=" * 60)
    
    # Create a safe environment for testing
    print("Creating safe environment...")
    env = make_real_world_env(
        config_type="safe",
        robot_ip="172.16.0.2",  # Replace with your robot IP
        episode_length_s=10.0   # Short episodes for testing
    )
    
    try:
        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset()
        
        print(f"Initial observation shape: {obs['policy'].shape}")
        print(f"Initial info: {info}")
        
        # Run a few steps with small random actions
        for step in range(10):
            # Small random actions for safety
            action = torch.randn(1, 7) * 0.02  # Very small actions
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step + 1:2d}: Reward={reward.item():6.3f}, "
                  f"Distance={info.get('distance_to_target', 0):.3f}, "
                  f"Done={terminated.item() or truncated.item()}")
            
            if terminated.item() or truncated.item():
                print("Episode completed!")
                break
        
        # Get safety report
        safety_report = env.get_safety_report()
        print(f"\nSafety Status: {safety_report['safety']['current_safety_level']}")
        print(f"Total Safety Events: {safety_report['safety']['total_safety_events']}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("This is expected if no physical robot is connected.")
        
    finally:
        env.close()


def example_configuration_options():
    """Demonstrate different configuration options."""
    print("\n" + "=" * 60)
    print("CONFIGURATION OPTIONS")
    print("=" * 60)
    
    configs = [
        ("Safe Configuration", "safe"),
        ("Standard Configuration", "standard"),
        ("Performance Configuration", "performance")
    ]
    
    for name, config_type in configs:
        print(f"\n{name}:")
        
        try:
            env = make_real_world_env(config_type, robot_ip="172.16.0.2")
            
            print(f"  Control frequency: {env.cfg.robot.control_frequency} Hz")
            print(f"  Action scale: {env.cfg.actions.action_scale}")
            print(f"  Episode length: {env.cfg.episode_length_s}s")
            print(f"  Workspace X: {env.cfg.robot.workspace_limits[:2]}")
            print(f"  Joint limits buffer: {env.cfg.robot.joint_limits_buffer}")
            
            env.close()
            
        except Exception as e:
            print(f"  Error creating config: {e}")
    
    # Custom configuration example
    print(f"\nCustom Configuration:")
    try:
        custom_env = make_real_world_env(
            "standard",
            robot_ip="192.168.1.100",           # Custom IP
            episode_length_s=25.0,              # Longer episodes
            robot_control_frequency=20.0,        # Lower frequency
            actions_action_scale=0.03,          # Smaller actions
            robot_joint_limits_buffer=0.15      # Larger safety buffer
        )
        
        print(f"  Custom IP: {custom_env.cfg.robot.robot_ip}")
        print(f"  Custom episode length: {custom_env.cfg.episode_length_s}s")
        print(f"  Custom control frequency: {custom_env.cfg.robot.control_frequency} Hz")
        
        custom_env.close()
        
    except Exception as e:
        print(f"  Error creating custom config: {e}")


def example_safety_monitoring():
    """Demonstrate safety monitoring features."""
    print("\n" + "=" * 60)
    print("SAFETY MONITORING")
    print("=" * 60)
    
    try:
        env = make_real_world_env("safe", robot_ip="172.16.0.2")
        
        # Register custom safety callbacks
        def safety_callback(event):
            print(f"Safety Event: {event.level.value} - {event.message}")
        
        from real_world_safety_monitor import SafetyLevel
        env._safety_monitor.register_callback(SafetyLevel.WARNING, safety_callback)
        env._safety_monitor.register_callback(SafetyLevel.CRITICAL, safety_callback)
        
        print("Safety callbacks registered")
        
        # Test action safety checking
        print("\nTesting action safety:")
        
        # Safe action
        safe_action = torch.tensor([[0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01]])
        is_safe = env._check_action_safety(safe_action)
        print(f"Safe action check: {is_safe}")
        
        # Unsafe action (too large)
        unsafe_action = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]])
        is_safe = env._check_action_safety(unsafe_action)  
        print(f"Unsafe action check: {is_safe}")
        
        # Get comprehensive safety report
        safety_report = env.get_safety_report()
        print(f"\nSafety Report Summary:")
        print(f"  Monitoring active: {safety_report['safety']['monitoring_active']}")
        print(f"  Emergency stop: {safety_report['safety']['emergency_stop_active']}")
        print(f"  Current level: {safety_report['safety']['current_safety_level']}")
        
        env.close()
        
    except Exception as e:
        print(f"Safety monitoring test error: {e}")


def example_policy_evaluation():
    """Example of evaluating a trained policy."""
    print("\n" + "=" * 60)
    print("POLICY EVALUATION")
    print("=" * 60)
    
    class SimplePolicy:
        """Simple reaching policy for demonstration."""
        
        def __init__(self, action_scale: float = 0.05):
            self.action_scale = action_scale
            
        def __call__(self, obs: torch.Tensor) -> torch.Tensor:
            """Simple policy that moves towards target."""
            # Extract components from observation
            # obs format: [progress, joint_pos(7), joint_vel(7), target_pos(3)]
            
            batch_size = obs.shape[0]
            
            # Simple random policy with small bias towards center
            action = torch.randn(batch_size, 7) * self.action_scale
            
            # Add small bias to move joints towards neutral position
            neutral_bias = torch.tensor([0, -0.1, 0, -0.1, 0, 0.1, 0]) * 0.1
            action += neutral_bias.unsqueeze(0)
            
            return action
    
    try:
        # Create environment
        env = make_real_world_env("standard", episode_length_s=15.0)
        policy = SimplePolicy(action_scale=0.03)
        
        # Evaluate policy for multiple episodes
        num_episodes = 3
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        print(f"Evaluating policy for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            print(f"\nEpisode {episode + 1}:")
            
            for step in range(100):  # Max steps per episode
                # Get action from policy
                action = policy(obs['policy'])
                
                # Execute step
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward.item()
                episode_length += 1
                
                # Log progress every 10 steps
                if step % 10 == 0:
                    distance = info.get('distance_to_target', 0)
                    print(f"  Step {step:2d}: Reward={reward.item():6.3f}, Distance={distance:.3f}")
                
                if terminated.item() or truncated.item():
                    success = info.get('success', False)
                    if success:
                        success_count += 1
                        print(f"  ✓ Target reached!")
                    else:
                        print(f"  Episode terminated/truncated")
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  Episode reward: {episode_reward:.3f}")
            print(f"  Episode length: {episode_length} steps")
        
        # Print evaluation summary
        print(f"\nEvaluation Summary:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
        print(f"  Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"  Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
        env.close()
        
    except Exception as e:
        print(f"Policy evaluation error: {e}")


def example_gymnasium_integration():
    """Demonstrate Gymnasium environment registration."""
    print("\n" + "=" * 60)
    print("GYMNASIUM INTEGRATION")
    print("=" * 60)
    
    try:
        # Register real-world environments  
        register_real_world_envs()
        
        # Use standard gym interface
        import gymnasium as gym
        
        print("Available real-world environments:")
        for env_id in ["RealWorld-Franka-Reach-Safe-v0", 
                      "RealWorld-Franka-Reach-v0",
                      "RealWorld-Franka-Reach-Performance-v0"]:
            print(f"  - {env_id}")
        
        # Create environment using gym.make
        env = gym.make("RealWorld-Franka-Reach-Safe-v0")
        
        print(f"\nCreated environment: {env}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        env.close()
        print("✓ Gymnasium integration successful")
        
    except ImportError:
        print("Gymnasium not available - skipping integration test")
    except Exception as e:
        print(f"Gymnasium integration error: {e}")


def example_training_loop():
    """Example training loop structure."""
    print("\n" + "=" * 60)
    print("TRAINING LOOP EXAMPLE")
    print("=" * 60)
    
    print("This is a template for integrating with training frameworks:")
    
    training_code = """
# Example training loop with real-world environment
from real_world_integration import make_real_world_env

def train_policy():
    # Create environment
    env = make_real_world_env("standard", robot_ip="YOUR_ROBOT_IP")
    
    # Initialize your policy/agent
    policy = YourPolicy(...)
    optimizer = YourOptimizer(...)
    
    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_data = []
            
            for step in range(max_steps):
                # Get action from policy
                action = policy(obs)
                
                # Execute step
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Store transition
                episode_data.append((obs, action, reward, next_obs, terminated))
                
                obs = next_obs
                
                if terminated.item() or truncated.item():
                    break
            
            # Update policy with episode data
            update_policy(policy, optimizer, episode_data)
            
            # Log progress
            if episode % 10 == 0:
                evaluate_policy(policy, env)
    
    finally:
        env.close()

# Framework-specific examples:

# RSL-RL Integration
from rsl_rl.runners import OnPolicyRunner
env = make_real_world_env("standard")
runner = OnPolicyRunner(env, cfg, log_dir="./logs")

# SKRL Integration  
from skrl.envs.torch import wrap_env
env = wrap_env(make_real_world_env("standard"))

# Stable-Baselines3 Integration
from stable_baselines3 import PPO
env = make_real_world_env("standard")
model = PPO("MlpPolicy", env, verbose=1)
"""
    
    print(training_code)


def main():
    """Run all examples."""
    print("Real-World Franka Robot Environment Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_configuration_options()
    example_safety_monitoring()
    example_policy_evaluation()
    example_gymnasium_integration()
    example_training_loop()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nTo run with a real robot:")
    print("1. Replace '172.16.0.2' with your robot's IP address")
    print("2. Ensure franky library is installed and robot is accessible")
    print("3. Start with 'safe' configuration for initial testing")
    print("4. Always keep emergency stop readily accessible")
    print("=" * 60)


if __name__ == "__main__":
    main()