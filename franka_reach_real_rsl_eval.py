#!/usr/bin/env python3

"""
RSL-RL evaluation script for Franka reaching task on real robot.
This script loads a trained RSL-RL model and evaluates it on the real Franka robot.
"""

import argparse
import os
import time
import torch
import numpy as np

# RSL-RL imports
from rsl_rl.runners import OnPolicyRunner

# Local imports
from franka_reach_pose_real import FrankaReachPose

def create_rsl_rl_config():
    """Create RSL-RL configuration for the real Franka environment."""
    
    config = {
        # Algorithm configuration
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "entropy_coef": 0.001,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 8,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "desired_kl": 0.01,
        },
        
        # Policy network configuration
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 1.0,
        },
        
        # Runner configuration
        "runner": {
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": 1000,
            "save_interval": 50,
            "experiment_name": "franka_reach_real",
            "run_name": "",
            "resume": False,
            "load_run": -1,
            "load_checkpoint": -1,
            "checkpoint_path": "",
        },
        
        # Environment configuration
        "env": {
            "num_envs": 1,  # Single real robot
            "episode_length_s": 30,  # 30 seconds per episode
        }
    }
    
    return config


class RealFrankaRSLWrapper:
    """Wrapper to make ReachingFranka compatible with RSL-RL."""
    
    def __init__(self, robot_ip="172.16.0.2", device="cuda:0"):
        self.env = FrankaReachPose(robot_ip=robot_ip, device=device)
        self.device = device
        self.num_envs = 1
        
        # RSL-RL expects these attributes
        self.num_obs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.max_episode_length = self.env.max_episode_length
        
    def reset(self):
        """Reset environment and return observations."""
        obs, info = self.env.reset()
        # RSL-RL expects observations in shape (num_envs, obs_dim)
        return torch.tensor(obs, device=self.device).unsqueeze(0)
    
    def step(self, actions):
        """Step environment with actions."""
        # Convert from tensor to numpy if needed
        if torch.is_tensor(actions):
            actions = actions.cpu().numpy().squeeze()
        
        obs, reward, terminated, truncated, info = self.env.step(actions)
        
        # Convert to tensors for RSL-RL
        obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
        reward_tensor = torch.tensor([reward], device=self.device)
        done_tensor = torch.tensor([terminated or truncated], device=self.device)
        
        return obs_tensor, reward_tensor, done_tensor, info
    
    def render(self, mode="human"):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate RSL-RL agent on real Franka robot")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to the trained RSL-RL checkpoint")
    parser.add_argument("--robot_ip", type=str, default="172.16.0.2",
                       help="IP address of the Franka robot")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run inference on")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--real_time", action="store_true", default=True,
                       help="Run evaluation in real-time")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic policy for evaluation")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print("=" * 80)
    print("RSL-RL Real Franka Robot Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Robot IP: {args.robot_ip}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    print("=" * 80)
    
    # Create the wrapped environment
    print("Initializing real robot environment...")
    try:
        env = RealFrankaRSLWrapper(robot_ip=args.robot_ip, device=args.device)
        print("✓ Real robot connection established")
    except Exception as e:
        print(f"✗ Failed to connect to robot: {e}")
        return
    
    # Load RSL-RL configuration
    config = create_rsl_rl_config()
    
    # Create RSL-RL runner
    print("Loading trained model...")
    try:
        log_dir = os.path.dirname(args.checkpoint)
        runner = OnPolicyRunner(env, config, log_dir=None, device=args.device)
        runner.load(args.checkpoint)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Get inference policy
    policy = runner.get_inference_policy(device=args.device)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print("\nStarting evaluation...")
    print("-" * 50)
    
    try:
        for episode in range(args.num_episodes):
            print(f"Episode {episode + 1}/{args.num_episodes}")
            
            # Reset environment
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Episode loop
            while True:
                start_time = time.time()
                
                # Get action from policy
                with torch.no_grad():
                    if args.deterministic:
                        action = policy(obs)
                    else:
                        action = policy.sample(obs)
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward.item()
                episode_length += 1
                
                # Check if episode is done
                if done.item() or episode_length >= env.max_episode_length:
                    # Check if target was reached (success condition)
                    if done.item() and episode_length < env.max_episode_length:
                        success_count += 1
                        print(f"  ✓ Target reached! Reward: {episode_reward:.3f}")
                    else:
                        print(f"  Episode timeout. Reward: {episode_reward:.3f}")
                    break
                
                # Real-time delay if requested
                if args.real_time:
                    elapsed = time.time() - start_time
                    sleep_time = env.env.dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  Length: {episode_length} steps")
            print("-" * 30)
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    
    except Exception as e:
        print(f"\nEvaluation error: {e}")
    
    finally:
        # Clean up
        env.close()
        print("\nRobot connection closed")
    
    # Print evaluation summary
    if episode_rewards:
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Success rate: {success_count}/{len(episode_rewards)} ({100*success_count/len(episode_rewards):.1f}%)")
        print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
        print(f"Best episode reward: {np.max(episode_rewards):.3f}")
        print(f"Worst episode reward: {np.min(episode_rewards):.3f}")
        print("=" * 50)


if __name__ == "__main__":
    main()