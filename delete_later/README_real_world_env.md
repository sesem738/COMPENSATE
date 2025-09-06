# Real-World Franka Robot Environment Interface

A comprehensive real-world environment interface for the Franka Emika robot that serves as an alternative to IsaacLab simulation environments. This package provides full compatibility with IsaacLab's ManagerBasedRLEnv interface while controlling a physical Franka robot using the franky library.

## Features

### 🤖 Robot Integration
- **Franky Library Integration**: Direct communication with physical Franka Panda robot
- **Multiple Control Modes**: Joint position, Cartesian, and impedance control
- **Real-time Control**: Configurable control frequencies (1-100 Hz)
- **Gripper Support**: Optional gripper control integration

### 🛡️ Safety System
- **Real-time Safety Monitoring**: Multi-threaded safety checks at 50 Hz
- **Emergency Stop**: Immediate robot halt on safety violations
- **Comprehensive Limit Checking**: Joint limits, workspace bounds, velocity/acceleration limits
- **Automatic Recovery**: Safe recovery procedures after emergency stops
- **Safety Event Logging**: Detailed logging and reporting of safety events

### ⚙️ Configuration Management
- **IsaacLab Compatible**: Mirrors IsaacLab configuration patterns
- **Predefined Configs**: Safe, standard, and performance configurations
- **Flexible Customization**: Easy parameter overrides and custom configurations
- **Validation System**: Automatic validation of configuration parameters

### 📊 Task Support
- **Reach Task**: End-effector reaching with configurable targets
- **Reward Shaping**: Distance-based rewards with success bonuses
- **Curriculum Learning**: Progressive difficulty adjustment
- **Performance Tracking**: Success rate, episode length, reward statistics

### 🔧 IsaacLab Integration
- **Drop-in Replacement**: Compatible with existing IsaacLab training scripts
- **Gymnasium Registration**: Standard gym environment registration
- **Tensor Operations**: PyTorch tensor support with GPU/CPU flexibility
- **Observation Compatibility**: Matches IsaacLab observation formats

## Installation

### Prerequisites
```bash
# Install franky library for Franka robot communication
pip install franky

# Install PyTorch (if not already installed)  
pip install torch

# Install Gymnasium (optional, for gym registration)
pip install gymnasium

# Install IsaacLab (optional, for full compatibility)
# Follow IsaacLab installation instructions
```

### Package Installation
```bash
# Clone or download the real-world environment files
# Ensure all Python files are in your Python path or working directory
```

## Quick Start

### Basic Usage
```python
from real_world_integration import make_real_world_env

# Create a safe environment for initial testing
env = make_real_world_env(
    config_type="safe",
    robot_ip="172.16.0.2"  # Replace with your robot's IP
)

# Reset environment
obs, info = env.reset()

# Run episode
for step in range(100):
    # Random action (replace with your policy)
    action = torch.randn(1, 7) * 0.1
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

### Using with IsaacLab Training Scripts
```python
# In your existing IsaacLab training script, simply replace:
# env = gym.make("Isaac-Reach-Franka-v0")

# With:
from real_world_integration import make_real_world_env
env = make_real_world_env("standard", robot_ip="YOUR_ROBOT_IP")

# Rest of your training script remains unchanged!
```

### Configuration Examples
```python
# Safe configuration (ultra-conservative for testing)
safe_env = make_real_world_env("safe")

# Standard configuration (normal operation)
standard_env = make_real_world_env("standard")

# Performance configuration (for trained policies)
perf_env = make_real_world_env("performance")

# Custom configuration
custom_env = make_real_world_env(
    "standard",
    robot_ip="192.168.1.100",
    episode_length_s=20.0,
    robot_control_frequency=25.0,
    actions_action_scale=0.05
)
```

## Architecture

### Core Components

1. **RealWorldFrankaEnv** (`real_world_env_interface.py`)
   - Main environment class with franky integration
   - Handles robot communication and control
   - Implements IsaacLab-compatible interface

2. **Configuration System** (`real_world_env_cfg.py`)  
   - IsaacLab-style configuration classes
   - Predefined safe/standard/performance configs
   - Parameter validation and consistency checks

3. **Safety Monitor** (`real_world_safety_monitor.py`)
   - Real-time safety monitoring in background thread
   - Emergency stop and recovery procedures
   - Comprehensive safety event logging

4. **Integration Wrapper** (`real_world_integration.py`)
   - Full IsaacLab ManagerBasedRLEnv compatibility
   - Gymnasium environment registration
   - Factory functions for easy environment creation

### Safety Features

The safety system provides multiple layers of protection:

- **Joint Limit Monitoring**: Prevents joint positions from exceeding safe ranges
- **Workspace Monitoring**: Ensures end-effector stays within defined workspace
- **Velocity/Acceleration Limits**: Prevents dangerous robot motions
- **Communication Monitoring**: Detects and handles connection issues
- **Emergency Stop**: Immediate robot halt with one command
- **Action Validation**: Pre-execution safety checks on all actions

## Configuration Options

### Robot Configuration
```python
robot: RealWorldRobotCfg(
    robot_ip="172.16.0.2",           # Robot IP address
    control_mode="joint_position",   # Control mode
    control_frequency=30.0,          # Control frequency (Hz)
    relative_dynamics_factor=0.2,    # Speed limiting (0.0-1.0)
    workspace_limits=[0.25, 0.8, -0.4, 0.4, 0.1, 0.8],  # [x_min, x_max, y_min, y_max, z_min, z_max]
)
```

### Action Configuration  
```python
actions: RealWorldActionsCfg(
    action_dim=7,                    # 7-DOF joint control
    action_scale=0.1,                # Action scaling factor
    use_normalized_actions=True,     # Actions in [-1, 1] range
    action_filter_enabled=True,      # Action smoothing
)
```

### Safety Configuration
```python
safety: RealWorldSafetyMonitorCfg(
    max_joint_velocity=2.0,          # rad/s
    max_end_effector_velocity=1.0,   # m/s
    joint_limit_margin=0.1,          # Safety buffer (radians)
    enforce_workspace_limits=True,   # Enable workspace checking
)
```

## Control Modes

### 1. Joint Position Control
- Direct control of individual joint positions
- Most precise control for complex motions
- Default and recommended mode

```python
env = make_real_world_env(
    config_type="standard",
    robot_control_mode="joint_position"
)
```

### 2. Cartesian Control  
- Control end-effector position and orientation
- Intuitive for reaching tasks
- Robot handles inverse kinematics

```python  
env = make_real_world_env(
    config_type="standard", 
    robot_control_mode="cartesian"
)
```

### 3. Impedance Control
- Force/torque-based compliant control
- Useful for contact-rich tasks
- Requires careful tuning

```python
env = make_real_world_env(
    config_type="standard",
    robot_control_mode="impedance"  
)
```

## Safety Guidelines

### Initial Testing
1. **Start with Safe Config**: Always begin with `config_type="safe"`
2. **Low Action Scale**: Use small action scales (0.02-0.05) initially
3. **Restricted Workspace**: Limit workspace to safe area away from obstacles
4. **Emergency Stop Ready**: Keep emergency stop button easily accessible

### Production Use
1. **Validate Policies**: Thoroughly test policies in simulation first
2. **Gradual Scaling**: Gradually increase action scales and workspace
3. **Monitor Safety Events**: Regularly check safety reports and logs
4. **Regular Calibration**: Periodically validate robot calibration

### Recovery Procedures
1. **Emergency Stop Reset**: Use `env.reset_emergency_stop()` after resolving issues
2. **Safety Report**: Check `env.get_safety_report()` for detailed status
3. **Manual Recovery**: Move robot to safe position if needed
4. **System Restart**: Restart environment if persistent issues occur

## Performance Optimization

### Control Frequency
- **Low Frequency (10-20 Hz)**: Safe for initial testing and development
- **Medium Frequency (20-30 Hz)**: Good balance for most applications  
- **High Frequency (30+ Hz)**: Maximum performance for trained policies

### Action Scaling
- **Conservative (0.02-0.05)**: Safe for testing and development
- **Standard (0.05-0.1)**: Normal operation with validated policies
- **Aggressive (0.1+)**: High performance with extensively tested policies

## Troubleshooting

### Connection Issues
```python
# Check robot connection
if not env._is_connected:
    print("Robot not connected - check IP address and network")

# Test basic connectivity
ping 172.16.0.2  # Replace with your robot IP
```

### Safety Violations
```python
# Get detailed safety report
report = env.get_safety_report()
print(f"Safety status: {report['safety']['current_safety_level']}")
print(f"Recent events: {report['safety']['recent_safety_events']}")

# Reset emergency stop if conditions are safe
if env._safety_monitor.emergency_stop_active:
    success = env._safety_monitor.reset_emergency_stop()
    print(f"Emergency stop reset: {success}")
```

### Performance Issues
```python
# Check control frequency vs actual frequency
# Monitor system resources
# Reduce action complexity if needed
# Consider lower control frequency for stability
```

## Integration Examples

### RSL-RL Training
```python
from real_world_integration import make_real_world_env
from rsl_rl.runners import OnPolicyRunner

# Create environment
env = make_real_world_env("standard")

# Use with RSL-RL (requires adaptation wrapper)
runner = OnPolicyRunner(env, cfg, log_dir="./logs")
runner.learn(num_learning_iterations=1000)
```

### SKRL Training  
```python
from real_world_integration import make_real_world_env
from skrl.envs.torch import wrap_env
from skrl.agents.torch.ppo import PPO

# Create and wrap environment
env = make_real_world_env("standard")  
env = wrap_env(env)

# Train with SKRL
agent = PPO(models=models, memory=memory, cfg=cfg, ...)
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
trainer.train()
```

### Custom Training Loop
```python
from real_world_integration import make_real_world_env

env = make_real_world_env("standard")

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        # Your policy here
        action = policy(obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: Reward = {episode_reward}")

env.close()
```

## License

This real-world environment interface is provided as-is for research and development purposes. Users are responsible for ensuring safe operation of their robotic systems.

## Contributing

Contributions are welcome! Please ensure all changes maintain safety as the top priority and include appropriate testing.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review safety guidelines and ensure proper robot setup
3. Verify franky library installation and robot connectivity
4. Check configuration parameters for consistency

Remember: Safety first! Always test thoroughly in simulation before deploying to physical robots.