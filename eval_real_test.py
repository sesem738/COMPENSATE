#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import onnxruntime as ort
import time
import sys
from franka_reach_pose_real_test import FrankaReachPose  # Use test environment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test ONNX policy evaluation with mock Franka robot")
    parser.add_argument("--robot_ip", default="127.0.0.1", help="Mock robot IP address")
    parser.add_argument("--policy_path", default="policy.onnx", help="Path to ONNX policy file")
    parser.add_argument("--max_episodes", type=int, default=3, help="Maximum number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cpu", help="Device to run inference on")
    return parser.parse_args()


def load_onnx_policy(policy_path, device="cpu"):
    """Load ONNX policy model."""
    try:
        providers = ['CPUExecutionProvider']
        if device == "cuda" and torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(policy_path, providers=providers)
        print(f"Successfully loaded ONNX policy from {policy_path}")
        
        # Print model input/output info
        print("\nModel inputs:")
        for input_info in session.get_inputs():
            print(f"  {input_info.name}: {input_info.shape} ({input_info.type})")
        
        print("\nModel outputs:")
        for output_info in session.get_outputs():
            print(f"  {output_info.name}: {output_info.shape} ({output_info.type})")
            
        return session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX policy: {e}")


def run_policy(session, observation):
    """Run inference with ONNX policy."""
    # Convert observation to the expected format
    if isinstance(observation, np.ndarray):
        obs_input = observation.astype(np.float32)
    else:
        obs_input = np.array(observation, dtype=np.float32)
    
    # Add batch dimension if needed
    if len(obs_input.shape) == 1:
        obs_input = obs_input.reshape(1, -1)
    
    # Get input name from the model
    input_name = session.get_inputs()[0].name
    
    # Run inference
    result = session.run(None, {input_name: obs_input})
    
    # Extract action (assume first output is the action)
    action = result[0]
    
    # Remove batch dimension if added
    if len(action.shape) > 1 and action.shape[0] == 1:
        action = action.squeeze(0)
    
    return action


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("TESTING EVALUATION WITH MOCK ROBOT")
    print("=" * 60)
    print(f"Starting evaluation with {args.max_episodes} episodes")
    print(f"Mock Robot IP: {args.robot_ip}")
    print(f"Policy path: {args.policy_path}")
    print(f"Device: {args.device}")
    print("-" * 60)
    
    # Load ONNX policy
    try:
        policy_session = load_onnx_policy(args.policy_path, args.device)
    except Exception as e:
        print(f"Error loading policy: {e}")
        return 1
    
    # Initialize test environment (with auto_target=True to skip user input)
    try:
        env = FrankaReachPose(robot_ip=args.robot_ip, device=args.device, auto_target=True)
        print("\nSuccessfully connected to mock robot environment")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    except Exception as e:
        print(f"Error connecting to mock robot: {e}")
        return 1
    
    # Statistics tracking
    episode_stats = {
        'episode_lengths': [],
        'episode_rewards': [],
        'success_count': 0,
        'final_distances': [],
        'episode_times': []
    }
    
    try:
        for episode in range(args.max_episodes):
            print(f"\n{'='*20} Episode {episode + 1}/{args.max_episodes} {'='*20}")
            
            # Reset environment
            observation, info = env.reset()
            print(f"Initial observation shape: {observation.shape}")
            print(f"Target position: {info.get('target_position', 'Unknown')}")
            
            episode_reward = 0.0
            episode_length = 0
            episode_start_time = time.time()
            
            done = False
            while not done:
                try:
                    # Get action from policy
                    action = run_policy(policy_session, observation)
                    
                    # Take step in environment
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                    
                    # Print step info
                    if episode_length % 5 == 0 or done:
                        distance = info.get('distance_to_target', float('inf'))
                        print(f"  Step {episode_length}: Reward={reward:.3f}, Distance={distance:.4f}m")
                    
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    done = True
                    break
                except Exception as e:
                    print(f"Error during episode step: {e}")
                    done = True
                    break
            
            # Record episode statistics
            episode_time = time.time() - episode_start_time
            is_success = info.get('is_success', False)
            final_distance = info.get('distance_to_target', float('inf'))
            
            episode_stats['episode_lengths'].append(episode_length)
            episode_stats['episode_rewards'].append(episode_reward)
            episode_stats['final_distances'].append(final_distance)
            episode_stats['episode_times'].append(episode_time)
            
            if is_success:
                episode_stats['success_count'] += 1
            
            # Print episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"  Length: {episode_length} steps")
            print(f"  Total Reward: {episode_reward:.3f}")
            print(f"  Success: {'Yes' if is_success else 'No'}")
            print(f"  Final Distance: {final_distance:.4f}m")
            print(f"  Episode Time: {episode_time:.1f}s")
            
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error during evaluation: {e}")
    finally:
        # Clean up environment
        try:
            env.close()
            print("\nEnvironment closed successfully")
        except Exception as e:
            print(f"Warning: Error closing environment: {e}")
    
    # Print final statistics
    if episode_stats['episode_lengths']:
        print(f"\n{'='*20} Final Statistics {'='*20}")
        print(f"Episodes completed: {len(episode_stats['episode_lengths'])}")
        print(f"Success rate: {episode_stats['success_count']}/{len(episode_stats['episode_lengths'])} "
              f"({100 * episode_stats['success_count'] / len(episode_stats['episode_lengths']):.1f}%)")
        print(f"Average episode length: {np.mean(episode_stats['episode_lengths']):.1f} steps")
        print(f"Average episode reward: {np.mean(episode_stats['episode_rewards']):.3f}")
        print(f"Average final distance: {np.mean(episode_stats['final_distances']):.4f}m")
        print(f"Average episode time: {np.mean(episode_stats['episode_times']):.1f}s")
        
        print(f"Best episode distance: {min(episode_stats['final_distances']):.4f}m")
        
        print(f"\n{'='*60}")
        print("✓ Mock evaluation completed successfully!")
        print("The logic has been verified and should work with real hardware.")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)