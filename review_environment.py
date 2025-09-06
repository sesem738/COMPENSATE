#!/usr/bin/env python3

import numpy as np
import sys
import os

# Mock the franky library for testing
sys.path.insert(0, '/home/silencio/WorldWideWeb/COMPENSATE')
import mock_franky
sys.modules['franky'] = mock_franky

from franka_reach_pose_real import FrankaReachPose

def review_environment():
    """Comprehensive review of the FrankaReachPose environment."""
    
    print("=" * 80)
    print("ENVIRONMENT REVIEW: franka_reach_pose_real.py")
    print("=" * 80)
    
    issues_found = []
    
    # Test 1: Environment initialization
    print("\n1. TESTING ENVIRONMENT INITIALIZATION")
    print("-" * 50)
    
    try:
        env = FrankaReachPose(auto_target=True)
        print(f"✓ Environment initialized successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Check observation buffer size mismatch
        print(f"  obs_buf size: {env.obs_buf.shape}")
        if env.obs_buf.shape[0] != env.observation_space.shape[0]:
            issue = f"obs_buf size ({env.obs_buf.shape[0]}) != observation_space size ({env.observation_space.shape[0]})"
            print(f"  ✗ ISSUE: {issue}")
            issues_found.append(issue)
        else:
            print(f"  ✓ obs_buf size matches observation_space")
            
    except Exception as e:
        issue = f"Environment initialization failed: {e}"
        print(f"✗ ISSUE: {issue}")
        issues_found.append(issue)
        return issues_found
    
    # Test 2: Reset functionality
    print("\n2. TESTING RESET FUNCTIONALITY")
    print("-" * 50)
    
    try:
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  Info keys: {list(info.keys())}")
        
        # Check if observation matches expected format
        if obs.shape[0] != 28:
            issue = f"Reset observation shape ({obs.shape[0]}) != expected (28)"
            print(f"  ✗ ISSUE: {issue}")
            issues_found.append(issue)
        else:
            print(f"  ✓ Observation shape matches policy expectation (28)")
            
    except Exception as e:
        issue = f"Reset failed: {e}"
        print(f"✗ ISSUE: {issue}")
        issues_found.append(issue)
    
    # Test 3: Step functionality
    print("\n3. TESTING STEP FUNCTIONALITY")
    print("-" * 50)
    
    try:
        # Test with random action
        action = env.action_space.sample()
        print(f"  Test action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info keys: {list(info.keys())}")
        
        # Check observation consistency
        if obs.shape[0] != 28:
            issue = f"Step observation shape ({obs.shape[0]}) != expected (28)"
            print(f"  ✗ ISSUE: {issue}")
            issues_found.append(issue)
            
    except Exception as e:
        issue = f"Step failed: {e}"
        print(f"✗ ISSUE: {issue}")
        issues_found.append(issue)
    
    # Test 4: Observation construction analysis
    print("\n4. ANALYZING OBSERVATION CONSTRUCTION")
    print("-" * 50)
    
    try:
        # Manually check the observation construction in _get_observation_reward_done
        obs, reward, done = env._get_observation_reward_done()
        print(f"  Raw observation shape: {obs.shape}")
        print(f"  env.obs_buf shape: {env.obs_buf.shape}")
        
        # Analyze observation components
        print(f"  Joint positions (0-6): {obs[:7]}")
        print(f"  Joint velocities (7-13): {obs[7:14]}")
        print(f"  Target position (14-20): {obs[14:21] if len(obs) > 20 else 'MISSING'}")
        print(f"  Previous actions (21-27): {obs[21:28] if len(obs) > 27 else 'MISSING'}")
        
        # Check if we're trying to access out-of-bounds indices
        if len(obs) < 28:
            issue = f"Observation too short: {len(obs)} < 28"
            print(f"  ✗ ISSUE: {issue}")
            issues_found.append(issue)
            
    except Exception as e:
        issue = f"Observation construction failed: {e}"
        print(f"✗ ISSUE: {issue}")
        issues_found.append(issue)
    
    # Test 5: Target generation analysis
    print("\n5. ANALYZING TARGET GENERATION")
    print("-" * 50)
    
    try:
        original_target = env.target_pos.copy()
        env._generate_random_target()
        new_target = env.target_pos
        
        print(f"  Original target: {original_target}")
        print(f"  Generated target: {new_target}")
        print(f"  Target shape: {new_target.shape}")
        
        # Check if target generation creates wrong shape
        if len(new_target) != 3:
            issue = f"Generated target has wrong shape: {new_target.shape} (expected 3D position)"
            print(f"  ✗ ISSUE: {issue}")
            issues_found.append(issue)
        else:
            print(f"  ✓ Target has correct 3D position shape")
            
    except Exception as e:
        issue = f"Target generation failed: {e}"
        print(f"✗ ISSUE: {issue}")
        issues_found.append(issue)
    
    # Test 6: Action handling
    print("\n6. TESTING ACTION HANDLING")
    print("-" * 50)
    
    try:
        # Test action storage bug
        test_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        original_previous_action = env.previous_action.copy()
        
        # Simulate step method's action handling
        env.step(test_action)
        
        print(f"  Test action: {test_action}")
        print(f"  Previous action before: {original_previous_action}")
        print(f"  Previous action after: {env.previous_action}")
        
        # Check if previous_action is updated correctly
        if not np.allclose(env.previous_action, test_action):
            issue = "previous_action not updated correctly in step()"
            print(f"  ✗ ISSUE: {issue}")
            issues_found.append(issue)
        else:
            print(f"  ✓ previous_action updated correctly")
            
    except Exception as e:
        issue = f"Action handling test failed: {e}"
        print(f"✗ ISSUE: {issue}")
        issues_found.append(issue)
    
    # Clean up
    try:
        env.close()
    except:
        pass
    
    # Summary
    print("\n" + "=" * 80)
    print("REVIEW SUMMARY")
    print("=" * 80)
    
    if issues_found:
        print(f"✗ {len(issues_found)} ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("✓ No issues found - environment appears to be working correctly")
    
    return issues_found

if __name__ == "__main__":
    issues = review_environment()
    sys.exit(len(issues))