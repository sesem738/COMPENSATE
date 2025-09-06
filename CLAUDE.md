# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COMPENSATE is a robotics research project focused on reinforcement learning for robotic reaching tasks using the Franka Panda robot. The codebase implements both simulation environments (Isaac Gym) and real robot control, with a focus on joint failure scenarios and fault-tolerant control.

## Architecture

### Core Components

1. **Simulation Environment** (`reaching_franka_isaacgym_env.py`)
   - Isaac Gym-based Franka reaching environment
   - Supports both joint and cartesian control spaces
   - Implements curriculum learning with joint failure scenarios
   - Uses SKRL framework for RL training

2. **Real Robot Environment** (`real_env/reach_env_franka.py`)
   - FrankaPy-based real robot control
   - Gym-compliant interface matching simulation
   - Supports waypoint and impedance control modes
   - Optional camera tracking integration

3. **Joint Failure System** (`joint_failure.py`)
   - Implements various joint failure types: complete, intermittent, fails_midway, works_midway
   - Curriculum learning support for progressive difficulty
   - Statistics tracking for failure distributions

4. **Training Scripts**
   - `reaching_franka_isaacgym_skrl_train.py` - SKRL-based training
   - `reaching_franka_isaacgym_skrl_eval.py` - Model evaluation
   - `run_policy.py` - Policy execution (configured for Cartpole but template for Franka)

### Directory Structure

- `FrankaPy_SKRL_Code/` - Main codebase
  - `asymmetric/` - Isaac Gym task configurations (YAML files)
  - `camera_rl/` - Computer vision and camera-based RL variants
  - `real_env/` - Real robot environment implementations
  - `reallib/` - Real robot utilities and base classes
  - `mujoco_env/` - MuJoCo environment (separate package)

## Key Configuration Files

### Isaac Gym Task Configuration
- `asymmetric/FrankaReach.yaml` - Basic reaching task parameters
- `asymmetric/FrankaReachPPOAsymmLSTM.yaml` - PPO training configuration with LSTM

### Important Parameters
- Control spaces: "joint" or "cartesian"
- Failure types: 'none', 'complete', 'intermittent', 'fails_midway', 'works_midway'
- Default episode length: 100 steps
- Default environment count: 512-1024 parallel environments

## Development Workflow

### Training a Model
```python
# Simulation training
python reaching_franka_isaacgym_skrl_train.py

# Evaluation
python reaching_franka_isaacgym_skrl_eval.py
```

### Real Robot Operation
```python
# Real robot reaching environment
python real_env/reach_env_franka.py
```

### Camera Utilities
Located in `camera_rl/`:
- `capture_photo.py` - Image capture utilities  
- `opencv_calibration.py` - Camera calibration
- `aruco_detection.py` - ArUco marker detection
- `color_detector.py` - Color-based object detection

## Dependencies

- **Isaac Gym** - Physics simulation
- **SKRL** - Reinforcement learning framework  
- **FrankaPy** - Real Franka robot control
- **PyTorch** - Deep learning backend
- **OpenCV** - Computer vision (camera_rl components)
- **Gymnasium** - RL environment interface

## Model Architecture

Default neural network:
- Policy: [256, 128, 64] hidden layers with ELU activation
- Value: [256, 128, 64] hidden layers with ELU activation  
- Optional LSTM layers for sequence modeling
- Gaussian policy for continuous control

## Testing

The codebase includes test environments in `mujoco_env/mujoco_env/envs/test_franka_env.py` for validating environment implementations.