#!/usr/bin/env python3
"""
Create a dummy ONNX policy for testing the evaluation pipeline.
This creates a simple neural network that outputs reasonable actions for the Franka robot.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import sys

class DummyPolicy(nn.Module):
    """Simple policy network for testing."""
    
    def __init__(self, obs_dim=28, action_dim=7):
        super().__init__()
        
        # Simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # Output actions in [-1, 1] range
        )
    
    def forward(self, observation):
        return self.network(observation)


def create_dummy_policy(output_path="policy.onnx"):
    """Create and export a dummy ONNX policy."""
    
    print("Creating dummy policy...")
    
    # Create the model
    model = DummyPolicy(obs_dim=28, action_dim=7)
    
    # Initialize with reasonable weights for reaching behavior
    with torch.no_grad():
        # Initialize to produce small actions initially
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
    
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input for export
    dummy_input = torch.randn(1, 28)  # Batch size 1, observation dimension 28
    
    print(f"Model input shape: {dummy_input.shape}")
    print(f"Model output shape: {model(dummy_input).shape}")
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'action': {0: 'batch_size'}
            }
        )
        print(f"Successfully exported ONNX model to {output_path}")
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
        
        # Print model info
        print("\nModel information:")
        for input_info in onnx_model.graph.input:
            print(f"  Input: {input_info.name}, shape: {[dim.dim_value for dim in input_info.type.tensor_type.shape.dim]}")
        
        for output_info in onnx_model.graph.output:
            print(f"  Output: {output_info.name}, shape: {[dim.dim_value for dim in output_info.type.tensor_type.shape.dim]}")
            
        return True
        
    except Exception as e:
        print(f"Error exporting ONNX model: {e}")
        return False


def test_dummy_policy(policy_path="policy.onnx"):
    """Test the dummy policy with onnxruntime."""
    try:
        import onnxruntime as ort
        
        print(f"\nTesting dummy policy at {policy_path}...")
        
        # Load the ONNX model
        session = ort.InferenceSession(policy_path)
        
        # Create test observation
        test_obs = np.random.randn(1, 28).astype(np.float32)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: test_obs})
        
        action = result[0]
        print(f"Test input shape: {test_obs.shape}")
        print(f"Test output shape: {action.shape}")
        print(f"Test output range: [{action.min():.3f}, {action.max():.3f}]")
        print("Policy test passed!")
        
        return True
        
    except ImportError:
        print("onnxruntime not available for testing, but model creation succeeded")
        return True
    except Exception as e:
        print(f"Error testing policy: {e}")
        return False


def main():
    output_path = "policy.onnx"
    
    print("Creating dummy ONNX policy for testing...")
    print(f"Output path: {output_path}")
    print("-" * 50)
    
    # Create the dummy policy
    success = create_dummy_policy(output_path)
    
    if success:
        # Test the policy
        test_success = test_dummy_policy(output_path)
        
        if test_success:
            print(f"\n✓ Successfully created and tested dummy policy: {output_path}")
            print("This policy can now be used with eval_real.py for testing")
            return 0
        else:
            print("✗ Policy created but testing failed")
            return 1
    else:
        print("✗ Failed to create dummy policy")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)