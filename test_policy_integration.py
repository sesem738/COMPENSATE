#!/usr/bin/env python3

import sys
import numpy as np
import onnxruntime as ort

# Mock the franky library for testing
sys.path.insert(0, '/home/silencio/WorldWideWeb/COMPENSATE')
import mock_franky
sys.modules['franky'] = mock_franky

from franka_reach_pose_real import FrankaReachPose

def test_policy_integration():
    """Test integration between fixed environment and actual policy.onnx"""
    
    print("=" * 80)
    print("TESTING POLICY INTEGRATION")
    print("=" * 80)
    
    # Load policy
    try:
        session = ort.InferenceSession("policy.onnx", providers=['CPUExecutionProvider'])
        print("✓ Policy loaded successfully")
        
        input_name = session.get_inputs()[0].name
        print(f"  Policy input: {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
        print(f"  Policy output: {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
        
    except Exception as e:
        print(f"✗ Failed to load policy: {e}")
        return False
    
    # Initialize environment
    try:
        env = FrankaReachPose(auto_target=True)
        print("✓ Environment initialized successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize environment: {e}")
        return False
    
    # Test complete episode
    try:
        print(f"\n--- Running test episode ---")
        obs, info = env.reset()
        print(f"✓ Reset successful, observation shape: {obs.shape}")
        
        episode_length = 0
        max_steps = 10  # Short test
        
        while episode_length < max_steps:
            # Get action from policy
            obs_batch = obs.reshape(1, -1).astype(np.float32)
            result = session.run(None, {input_name: obs_batch})
            action = result[0].squeeze()
            
            print(f"  Step {episode_length + 1}:")
            print(f"    Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
            print(f"    Action range: [{action.min():.6f}, {action.max():.6f}]")
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            
            print(f"    Reward: {reward:.4f}")
            print(f"    Distance: {info.get('distance_to_target', 0):.4f}m")
            
            if terminated or truncated:
                print(f"  Episode terminated after {episode_length} steps")
                break
        
        print(f"✓ Episode completed successfully")
        
    except Exception as e:
        print(f"✗ Episode failed: {e}")
        return False
    finally:
        try:
            env.close()
        except:
            pass
    
    print(f"\n✓ Policy integration test passed!")
    return True

def test_eval_real_script():
    """Test the eval_real.py script with fixed environment"""
    
    print("\n" + "=" * 80)
    print("TESTING eval_real.py SCRIPT")
    print("=" * 80)
    
    # Create a test version of eval_real.py that uses mock robot
    test_script = '''
import sys
sys.path.insert(0, "/home/silencio/WorldWideWeb/COMPENSATE")

# Replace franky with mock
import mock_franky
sys.modules['franky'] = mock_franky

# Import eval_real components
from eval_real import load_onnx_policy, run_policy, main
import argparse

# Mock args
class TestArgs:
    robot_ip = "127.0.0.1"
    policy_path = "policy.onnx"
    max_episodes = 2
    seed = 42
    device = "cpu"
    auto_target = True

def test_parse_args():
    return TestArgs()

# Patch the parse_args
import eval_real
eval_real.parse_args = test_parse_args

if __name__ == "__main__":
    exit_code = main()
    print(f"Exit code: {exit_code}")
'''
    
    with open("test_eval_real_fixed.py", "w") as f:
        f.write(test_script)
    
    try:
        import subprocess
        result = subprocess.run(["python", "test_eval_real_fixed.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ eval_real.py script test passed")
            if "Successfully loaded ONNX policy" in result.stdout:
                print("  ✓ Policy loading confirmed")
            if "Episode" in result.stdout and "Summary" in result.stdout:
                print("  ✓ Episode execution confirmed")  
            if "Final Statistics" in result.stdout:
                print("  ✓ Statistics collection confirmed")
            return True
        else:
            print("✗ eval_real.py script test failed")
            print(f"  Return code: {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[:500]}")
            return False
            
    except Exception as e:
        print(f"✗ Error running eval_real.py test: {e}")
        return False
    finally:
        import os
        if os.path.exists("test_eval_real_fixed.py"):
            os.remove("test_eval_real_fixed.py")

if __name__ == "__main__":
    success1 = test_policy_integration()
    success2 = test_eval_real_script()
    
    print(f"\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    if success1 and success2:
        print("✓ ALL TESTS PASSED!")
        print("The fixed environment is ready for deployment with policy.onnx")
        exit_code = 0
    else:
        print("✗ Some tests failed")
        exit_code = 1
        
    sys.exit(exit_code)