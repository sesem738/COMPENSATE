#!/usr/bin/env python3

import sys
import os

# Mock the franky library for testing
sys.path.insert(0, '/home/silencio/WorldWideWeb/COMPENSATE')
import mock_franky
sys.modules['franky'] = mock_franky

# Now we can import the real modules
from franka_reach_pose_real import FrankaReachPose
from eval_real import load_onnx_policy, run_policy
import numpy as np

def test_eval_real_direct():
    """Direct test of eval_real functionality with fixed environment"""
    
    print("=" * 60)
    print("DIRECT eval_real.py TEST")
    print("=" * 60)
    
    # Test parameters
    robot_ip = "127.0.0.1"
    policy_path = "policy.onnx"
    max_episodes = 2
    seed = 42
    device = "cpu"
    
    np.random.seed(seed)
    
    try:
        # Load policy
        policy_session = load_onnx_policy(policy_path, device)
        print("✓ Policy loaded successfully")
        
        # Initialize environment with auto_target=True
        env = FrankaReachPose(robot_ip=robot_ip, device=device, auto_target=True)
        print("✓ Environment initialized with auto_target=True")
        
        # Run episodes
        episode_stats = {
            'episode_lengths': [],
            'episode_rewards': [],
            'success_count': 0,
            'final_distances': []
        }
        
        for episode in range(max_episodes):
            print(f"\n--- Episode {episode + 1}/{max_episodes} ---")
            
            # Reset
            observation, info = env.reset()
            print(f"Reset: target={info.get('target_position', 'unknown')}")
            
            episode_reward = 0.0
            episode_length = 0
            max_steps = 20  # Shorter for testing
            
            while episode_length < max_steps:
                # Get action from policy
                action = run_policy(policy_session, observation)
                
                # Step
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if episode_length % 5 == 0:
                    print(f"  Step {episode_length}: reward={reward:.3f}, distance={info.get('distance_to_target', 0):.3f}m")
                
                if terminated or truncated:
                    break
            
            # Record stats
            is_success = info.get('is_success', False)
            final_distance = info.get('distance_to_target', float('inf'))
            
            episode_stats['episode_lengths'].append(episode_length)
            episode_stats['episode_rewards'].append(episode_reward)
            episode_stats['final_distances'].append(final_distance)
            if is_success:
                episode_stats['success_count'] += 1
            
            print(f"Episode {episode + 1} complete: length={episode_length}, reward={episode_reward:.2f}, success={is_success}")
        
        # Summary
        print(f"\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Episodes: {len(episode_stats['episode_lengths'])}")
        print(f"Success rate: {episode_stats['success_count']}/{len(episode_stats['episode_lengths'])} ({100 * episode_stats['success_count'] / len(episode_stats['episode_lengths']):.1f}%)")
        print(f"Avg episode length: {np.mean(episode_stats['episode_lengths']):.1f}")
        print(f"Avg reward: {np.mean(episode_stats['episode_rewards']):.3f}")
        print(f"Avg final distance: {np.mean(episode_stats['final_distances']):.3f}m")
        
        env.close()
        print("✓ Direct eval_real test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Direct eval_real test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_eval_real_direct()
    sys.exit(0 if success else 1)