#!/usr/bin/env python3
"""
Test script to verify the complete evaluation pipeline works correctly.
This tests the integration between the environment, ONNX policy loading, and evaluation loop
using mock robot interfaces.
"""

import sys
import os
import subprocess
import numpy as np
import time
from pathlib import Path


def run_command(cmd, capture_output=True, timeout=30):
    """Run a command and return success status and output."""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, timeout=timeout)
            return result.returncode == 0, "", ""
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_dummy_policy_creation():
    """Test creation of dummy ONNX policy."""
    print("1. Testing dummy policy creation...")
    
    # Remove existing policy if it exists
    if os.path.exists("policy.onnx"):
        os.remove("policy.onnx")
    
    success, stdout, stderr = run_command("python create_dummy_policy.py")
    
    if success and os.path.exists("policy.onnx"):
        print("   ✓ Dummy policy created successfully")
        return True
    else:
        print("   ✗ Failed to create dummy policy")
        if stderr:
            print(f"   Error: {stderr}")
        return False


def test_mock_environment():
    """Test the mock environment can be imported and initialized."""
    print("2. Testing mock environment...")
    
    try:
        # Test importing the test environment
        from franka_reach_pose_real_test import FrankaReachPose
        
        print("   ✓ Successfully imported test environment")
        
        # Test environment initialization
        env = FrankaReachPose(auto_target=True)
        print(f"   ✓ Environment initialized")
        print(f"   ✓ Observation space: {env.observation_space}")
        print(f"   ✓ Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"   ✓ Reset successful, observation shape: {obs.shape}")
        
        # Test step
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        print(f"   ✓ Step successful, reward: {reward:.3f}")
        
        env.close()
        print("   ✓ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Mock environment test failed: {e}")
        return False


def test_onnx_policy_loading():
    """Test ONNX policy loading and inference."""
    print("3. Testing ONNX policy loading...")
    
    try:
        import onnxruntime as ort
        
        if not os.path.exists("policy.onnx"):
            print("   ✗ policy.onnx not found")
            return False
        
        # Load ONNX model
        session = ort.InferenceSession("policy.onnx", providers=['CPUExecutionProvider'])
        print("   ✓ ONNX policy loaded successfully")
        
        # Test inference
        test_obs = np.random.randn(1, 28).astype(np.float32)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: test_obs})
        action = result[0]
        
        print(f"   ✓ Policy inference successful")
        print(f"   ✓ Input shape: {test_obs.shape}, Output shape: {action.shape}")
        print(f"   ✓ Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        return True
        
    except ImportError:
        print("   ✗ onnxruntime not available")
        return False
    except Exception as e:
        print(f"   ✗ ONNX policy loading failed: {e}")
        return False


def test_eval_real_script():
    """Test the eval_real.py script with mock environment."""
    print("4. Testing eval_real.py script...")
    
    # Create a modified version of eval_real.py that uses the test environment
    test_script_content = '''
import sys
sys.path.insert(0, "/home/silencio/WorldWideWeb/COMPENSATE")

# Monkey patch to use test environment
import franka_reach_pose_real_test
sys.modules['franka_reach_pose_real'] = franka_reach_pose_real_test

# Now import and run the real eval script
from eval_real import main, parse_args
import argparse

# Override args for testing
class TestArgs:
    robot_ip = "127.0.0.1"
    policy_path = "policy.onnx"
    max_episodes = 2
    seed = 42
    device = "cpu"
    auto_target = True

# Monkey patch parse_args to return test args
def test_parse_args():
    return TestArgs()

# Replace the parse_args function
import eval_real
eval_real.parse_args = test_parse_args

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
'''
    
    # Write the test script
    with open("test_eval_real_runner.py", "w") as f:
        f.write(test_script_content)
    
    try:
        # Run the test script
        success, stdout, stderr = run_command("python test_eval_real_runner.py", timeout=60)
        
        if success:
            print("   ✓ eval_real.py script ran successfully")
            # Check for key indicators in the output
            if "Successfully loaded ONNX policy" in stdout:
                print("   ✓ Policy loading confirmed")
            if "Episode" in stdout and "Summary" in stdout:
                print("   ✓ Episode execution confirmed")
            if "Final Statistics" in stdout:
                print("   ✓ Statistics collection confirmed")
            return True
        else:
            print("   ✗ eval_real.py script failed")
            if stderr:
                print(f"   Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error running eval_real.py: {e}")
        return False
    finally:
        # Clean up test script
        if os.path.exists("test_eval_real_runner.py"):
            os.remove("test_eval_real_runner.py")


def test_shape_compatibility():
    """Test that observation and action shapes are compatible."""
    print("5. Testing shape compatibility...")
    
    try:
        from franka_reach_pose_real_test import FrankaReachPose
        import onnxruntime as ort
        
        # Initialize environment
        env = FrankaReachPose(auto_target=True)
        
        # Load policy
        session = ort.InferenceSession("policy.onnx", providers=['CPUExecutionProvider'])
        
        # Get observation from environment
        obs, info = env.reset()
        print(f"   Environment observation shape: {obs.shape}")
        
        # Check policy expects correct input shape
        input_shape = session.get_inputs()[0].shape
        expected_obs_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        print(f"   Policy expected input shape: {input_shape}")
        
        if obs.shape[0] == expected_obs_dim:
            print("   ✓ Observation shapes are compatible")
        else:
            print(f"   ✗ Shape mismatch: env={obs.shape[0]}, policy={expected_obs_dim}")
            env.close()
            return False
        
        # Test policy inference with environment observation
        obs_batch = obs.reshape(1, -1).astype(np.float32)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: obs_batch})
        action = result[0].squeeze()
        
        print(f"   Policy output shape: {action.shape}")
        print(f"   Environment action space shape: {env.action_space.shape}")
        
        if action.shape == env.action_space.shape:
            print("   ✓ Action shapes are compatible")
        else:
            print(f"   ✗ Action shape mismatch: policy={action.shape}, env={env.action_space.shape}")
            env.close()
            return False
        
        # Test that action is within expected bounds
        if np.all(action >= -4) and np.all(action <= 4):
            print("   ✓ Action values within expected bounds")
        else:
            print(f"   ! Action values outside expected bounds: {action}")
            print("   (This may be OK depending on the policy)")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ✗ Shape compatibility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING EVALUATION PIPELINE")
    print("=" * 60)
    
    tests = [
        test_dummy_policy_creation,
        test_mock_environment, 
        test_onnx_policy_loading,
        test_shape_compatibility,
        test_eval_real_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"   ✗ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! The evaluation pipeline is working correctly.")
        print("\nYou can now run:")
        print("  python eval_real.py --max_episodes 5 --auto_target")
        print("to test the complete system with the mock robot.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)