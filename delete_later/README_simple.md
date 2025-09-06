# Simple Real-World Franka Environment

A simplified real-world environment interface for the Franka Emika robot, following the style and structure of the original `franka_reach_real_env.py`. This provides an easy-to-use alternative to IsaacLab simulation with minimal complexity.

## Features

✅ **Simple & Clean**: Follows the original `franka_reach_real_env.py` structure  
✅ **IsaacLab Compatible**: Drop-in replacement with tensor support  
✅ **Basic Safety**: Essential safety checks and limits  
✅ **Multiple Configs**: Safe, standard, performance, and test presets  
✅ **Auto-Target Mode**: No user input required for automated training  
✅ **Gymnasium Integration**: Standard gym environment registration

## Quick Start

### Basic Usage
```python
from franka_real_simple import make_franka_real_env

# Create environment (automatically generates targets)
env = make_franka_real_env(
    robot_ip="172.16.0.2",  # Your robot's IP
    config="safe",          # Safe configuration
    auto_target=True        # No user input needed
)

# Standard gym interface
obs, info = env.reset()

for step in range(100):
    action = your_policy(obs)  # Your trained policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        break

env.close()
```

### IsaacLab Compatibility
```python
# Get IsaacLab-compatible wrapper with tensor I/O
env = make_franka_real_env(
    config="standard",
    isaaclab_wrapper=True  # Returns tensors, batch dimensions
)

obs_dict, info = env.reset()  # obs_dict["policy"] is torch tensor
actions = torch.randn(1, 7)  # Batch dimension required
obs_dict, reward, terminated, truncated, info = env.step(actions)
```

### Training Integration
```python
# Works with any training framework
env = make_franka_real_env(config="standard", auto_target=True)

# Your existing training loop
for episode in range(num_episodes):
    obs, info = env.reset()
    
    while True:
        action = policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Update your policy here
        
        if done or truncated:
            break
```

## Configuration Options

### Predefined Configs

**Safe** (recommended for first use):
- Very slow robot motion (10% speed)
- Small actions (0.05 scale)
- Restricted workspace 
- Conservative safety limits

**Standard** (normal operation):
- Moderate robot motion (15% speed)  
- Normal actions (0.1 scale)
- Full workspace access
- Balanced safety/performance

**Performance** (for trained policies):
- Faster robot motion (25% speed)
- Larger actions (0.15 scale)
- Extended workspace
- Optimized for speed

**Test** (automated testing):
- Very safe settings
- Short episodes (100 steps)
- Automatic target generation
- No user interaction required

### Usage Examples
```python
# Ultra-safe for first tests
env = make_franka_real_env(config="safe")

# Normal operation  
env = make_franka_real_env(config="standard")

# High performance
env = make_franka_real_env(config="performance")

# Automated testing
env = make_franka_real_env(config="test", auto_target=True)
```

## File Structure

### Core Files (simplified version)
- `franka_real_simple.py` - Complete integrated environment
- `franka_real_config.py` - Configuration system (optional)
- `franka_real_safety.py` - Safety utilities (optional)

### Complete Files (full version)  
- `real_world_env_interface.py` - Detailed environment class
- `real_world_env_cfg.py` - Comprehensive configuration 
- `real_world_safety_monitor.py` - Advanced safety system
- `real_world_integration.py` - Full IsaacLab integration

## Key Differences from Original

| Feature | Original `franka_reach_real_env.py` | Simplified Version |
|---------|-----------------------------------|-------------------|
| Target Input | Always asks user for target | `auto_target=True` option |
| Configuration | Hardcoded parameters | Multiple preset configs |
| Safety | Basic error handling | Integrated safety checks |
| IsaacLab | Not compatible | Optional wrapper provided |
| Gym Registration | None | Automatic registration |
| Action Safety | Basic joint limits | Action magnitude + prediction |

## Safety Features

The simplified environment includes essential safety features:

- **Joint limit checking** with safety buffers
- **Action magnitude limiting** to prevent large motions  
- **Workspace boundary enforcement**
- **Predicted position validation** before motion execution
- **Emergency termination** on safety violations
- **Automatic scaling** of unsafe actions

## Installation & Setup

```bash
# Install required packages
pip install franky torch gymnasium numpy

# Ensure robot is powered on and network accessible
ping YOUR_ROBOT_IP

# Test connection (replace IP)
python -c "from franky import Robot; Robot('172.16.0.2')"
```

## Usage Patterns

### 1. Interactive Development
```python
# Manual target input for testing specific positions
env = make_franka_real_env(auto_target=False)
obs, info = env.reset()  # Will ask for target input
```

### 2. Automated Training
```python  
# No user interaction required
env = make_franka_real_env(auto_target=True, config="safe")

for episode in range(1000):
    obs, info = env.reset()  # Random target generated
    # ... training loop
```

### 3. Policy Evaluation
```python
# Test trained policy with multiple random targets
env = make_franka_real_env(config="performance", auto_target=True)

success_rate = evaluate_policy(policy, env, num_episodes=50)
```

### 4. Gymnasium Integration
```python
import gymnasium as gym

# Use standard gym interface
env = gym.make("FrankaReal-Safe-v0")  # Auto-registered
obs, info = env.reset()
```

## Troubleshooting

### Connection Issues
```python
# Test robot connection
from franka_real_safety import check_robot_ready
from franky import Robot

robot = Robot("YOUR_IP")
is_ready, msg = check_robot_ready(robot)
print(f"Robot ready: {is_ready} - {msg}")
```

### Safety Warnings
The environment will automatically:
- Scale down large actions
- Prevent joint limit violations  
- Warn about workspace violations
- Terminate episodes on safety issues

### Common Issues
1. **"Motion failed" errors**: Usually network latency - try lower control frequency
2. **Joint limit warnings**: Action scale too large - use smaller config
3. **Workspace violations**: Target outside reachable area - check workspace limits
4. **Connection timeouts**: Robot not responsive - check network and robot status

## Example Training Script

```python
from franka_real_simple import make_franka_real_env
import numpy as np

def train_simple_policy():
    # Create environment
    env = make_franka_real_env(
        robot_ip="172.16.0.2",  # Update this!
        config="safe",          # Start safe
        auto_target=True        # No user input
    )
    
    # Simple random policy for demonstration
    for episode in range(10):
        obs, info = env.reset()
        episode_reward = 0
        
        print(f"Episode {episode + 1}, Target: {info['target_position']}")
        
        for step in range(100):
            # Random action (replace with your policy)
            action = np.random.randn(7) * 0.05
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if step % 10 == 0:
                print(f"  Step {step}: Distance {info['distance_to_target']:.3f}")
            
            if terminated or truncated:
                success = info.get('is_success', False)
                print(f"  Episode finished: Success={success}, Reward={episode_reward:.2f}")
                break
    
    env.close()
    print("Training completed!")

if __name__ == "__main__":
    train_simple_policy()
```

## Migration from Original

To migrate from `franka_reach_real_env.py`:

```python
# Old way
from franka_reach_real_env import ReachingFranka
env = ReachingFranka(robot_ip="172.16.0.2")

# New way  
from franka_real_simple import make_franka_real_env
env = make_franka_real_env(
    robot_ip="172.16.0.2",
    auto_target=True,  # Avoid manual target input
    config="standard"
)
```

Key improvements:
- No more manual target input (unless desired)
- Built-in safety checks
- Multiple configuration options
- IsaacLab compatibility
- Better error handling
- Gymnasium registration

## Safety Guidelines

1. **Always start with `config="safe"`** for new setups
2. **Keep emergency stop accessible** during operation
3. **Test in simulation first** before using real robot
4. **Use `auto_target=True`** to avoid manual intervention
5. **Monitor robot motion** during initial testing
6. **Gradually increase performance** after validation

The simplified environment maintains the ease of use from the original while adding essential features for practical deployment.