# Critical Issues Found in franka_reach_pose_real.py

## Summary
Found 6 critical issues that prevent the environment from working with the actual `policy.onnx` file. These must be fixed before deployment.

## Issues Identified

### 1. **Observation Buffer Size Mismatch** ⚠️ CRITICAL
- **Problem**: `obs_buf` initialized with shape `(18,)` but `observation_space` expects `(28,)`
- **Location**: Line 54 in `franka_reach_pose_real.py`
- **Current**: `self.obs_buf = np.zeros((18,), dtype=np.float32)`
- **Should be**: `self.obs_buf = np.zeros((28,), dtype=np.float32)`
- **Impact**: Will cause array indexing errors when building observations

### 2. **Target Generation Wrong Shape** ⚠️ CRITICAL 
- **Problem**: `_generate_random_target()` creates 7D target instead of 3D position
- **Location**: Lines 108-116 in `franka_reach_pose_real.py`
- **Current**: Returns `[x, y, z, q_w, q_x, q_y, q_z]` (7 elements)
- **Should be**: Returns `[x, y, z]` (3 elements for position only)
- **Impact**: Causes shape broadcasting errors throughout the code

### 3. **Observation Construction Array Broadcasting** ⚠️ CRITICAL
- **Problem**: Trying to assign 3D target position to 7D slice in observation
- **Location**: Line 78 in `franka_reach_pose_real.py` 
- **Current**: `self.obs_buf[14:21] = self.target_pos` (expects 7 elements, gets 3 or 7)
- **Root cause**: Related to target generation shape issue
- **Impact**: Runtime errors during observation construction

### 4. **Action Storage Bug** ⚠️ MODERATE
- **Problem**: `previous_action` not updated correctly in `step()` method
- **Location**: Line 201 in `franka_reach_pose_real.py`
- **Current**: `action = self.previous_action` (overwrites input action)
- **Should be**: `self.previous_action = action.copy()` (store the input action)
- **Impact**: Policy actions not properly tracked in observations

### 5. **Auto-Target Mode Missing** ⚠️ MODERATE
- **Problem**: Environment always prompts for user input during reset
- **Location**: Lines 145-158 in `franka_reach_pose_real.py`
- **Current**: Always uses `input()` which blocks automated evaluation
- **Should be**: Respect `auto_target` parameter to skip user input
- **Impact**: Cannot run automated evaluation scripts

### 6. **Distance Calculation Redundancy** ⚠️ MINOR
- **Problem**: Distance to target calculated twice with potential inconsistency
- **Location**: Lines 82 and 206-209 in `franka_reach_pose_real.py`
- **Impact**: Minor performance issue, potential for inconsistent values

## Policy Requirements (from analysis)
The actual `policy.onnx` expects:
- **Input**: `(batch_size, 28)` float32 tensor
- **Output**: `(batch_size, 7)` float32 tensor  
- **Action range**: Approximately `[-0.001, 0.001]` (very small values)

## Expected Observation Format (28D)
Based on standard IsaacLab format:
1. Joint positions relative to default (7D): `robot_dof_pos - robot_default_dof_pos`
2. Joint velocities scaled (7D): `robot_dof_vel * dof_vel_scale`  
3. Target position (3D): `target_pos`
4. Previous action (7D): `previous_action`
5. Additional state (4D): Could be end-effector position + progress

## Proposed Fixes
1. Fix observation buffer size to 28D
2. Fix target generation to return only 3D position
3. Restructure observation construction for 28D format
4. Fix action storage in step method
5. Add auto_target mode support
6. Clean up distance calculation