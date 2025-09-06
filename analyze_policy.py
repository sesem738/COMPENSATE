#!/usr/bin/env python3

import onnx
import onnxruntime as ort
import numpy as np

def analyze_policy(policy_path="policy.onnx"):
    """Analyze the policy.onnx file structure and test inference."""
    
    print("=" * 60)
    print(f"ANALYZING POLICY: {policy_path}")
    print("=" * 60)
    
    try:
        # Load with ONNX for structure analysis
        model = onnx.load(policy_path)
        print(f"IR Version: {model.ir_version}")
        print(f"Opset Version: {model.opset_import[0].version}")
        
        print("\nModel Inputs:")
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value != 0 else "dynamic" for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {shape} (type: {inp.type.tensor_type.elem_type})")
        
        print("\nModel Outputs:")
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value != 0 else "dynamic" for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {shape} (type: {out.type.tensor_type.elem_type})")
            
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False
    
    try:
        # Load with ONNX Runtime for inference testing
        print("\n" + "=" * 60)
        print("TESTING ONNX RUNTIME INFERENCE")
        print("=" * 60)
        
        session = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
        
        print("\nRuntime Input Info:")
        for inp in session.get_inputs():
            print(f"  {inp.name}: {inp.shape} ({inp.type})")
            
        print("\nRuntime Output Info:")
        for out in session.get_outputs():
            print(f"  {out.name}: {out.shape} ({out.type})")
        
        # Test inference with different observation sizes
        test_shapes = [18, 28, 32]  # Common observation dimensions
        
        for obs_dim in test_shapes:
            print(f"\n--- Testing with observation dimension {obs_dim} ---")
            try:
                test_obs = np.random.randn(1, obs_dim).astype(np.float32)
                input_name = session.get_inputs()[0].name
                
                result = session.run(None, {input_name: test_obs})
                action = result[0]
                
                print(f"  ✓ Input shape: {test_obs.shape}")
                print(f"  ✓ Output shape: {action.shape}")
                print(f"  ✓ Action range: [{action.min():.4f}, {action.max():.4f}]")
                print(f"  ✓ Action values: {action.flatten()[:5]}...")  # Show first 5 values
                
                return True, obs_dim, action.shape[1] if len(action.shape) > 1 else action.shape[0]
                
            except Exception as e:
                print(f"  ✗ Failed with {obs_dim}D input: {e}")
        
        return False, None, None
        
    except Exception as e:
        print(f"Error with ONNX Runtime: {e}")
        return False, None, None

if __name__ == "__main__":
    success, obs_dim, action_dim = analyze_policy()
    
    if success:
        print(f"\n✓ Policy analysis successful!")
        print(f"Expected observation dimension: {obs_dim}")
        print(f"Expected action dimension: {action_dim}")
    else:
        print(f"\n✗ Policy analysis failed!")