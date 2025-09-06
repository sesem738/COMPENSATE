"""
Basic safety utilities for real-world Franka robot environment.
Simple safety checks following the style of the original franka_reach_real_env.py.
"""

import numpy as np
import time
from typing import Tuple, Optional


class FrankaSafetyChecker:
    """Simple safety checker for real-world Franka operations."""
    
    def __init__(self):
        # Standard Franka joint limits
        self.joint_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Safety parameters
        self.joint_buffer = 0.1  # radians safety buffer
        self.max_joint_velocity = 2.0  # rad/s
        self.max_ee_velocity = 1.0  # m/s
        self.max_action_norm = 0.5  # maximum action magnitude
        
        # Workspace limits (conservative defaults)
        self.workspace_limits = {
            'x': (0.3, 0.8),
            'y': (-0.4, 0.4), 
            'z': (0.1, 0.8)
        }
        
        # State tracking for velocity estimation
        self._prev_joint_pos = None
        self._prev_ee_pos = None
        self._prev_time = None
        
    def check_joint_limits(self, joint_positions: np.ndarray, buffer: Optional[float] = None) -> Tuple[bool, str]:
        """Check if joint positions are within safe limits.
        
        Args:
            joint_positions: Current joint positions
            buffer: Optional safety buffer override
            
        Returns:
            Tuple of (is_safe, warning_message)
        """
        if buffer is None:
            buffer = self.joint_buffer
            
        safe_lower = self.joint_lower_limits + buffer
        safe_upper = self.joint_upper_limits - buffer
        
        violations = []
        for i, (pos, lower, upper) in enumerate(zip(joint_positions, safe_lower, safe_upper)):
            if pos < lower:
                violations.append(f"Joint {i+1} too low: {pos:.3f} < {lower:.3f}")
            elif pos > upper:
                violations.append(f"Joint {i+1} too high: {pos:.3f} > {upper:.3f}")
        
        if violations:
            return False, "; ".join(violations)
        return True, ""
    
    def check_workspace_limits(self, ee_position: np.ndarray) -> Tuple[bool, str]:
        """Check if end effector is within workspace limits.
        
        Args:
            ee_position: End effector position [x, y, z]
            
        Returns:
            Tuple of (is_safe, warning_message)  
        """
        x, y, z = ee_position
        violations = []
        
        x_min, x_max = self.workspace_limits['x']
        y_min, y_max = self.workspace_limits['y'] 
        z_min, z_max = self.workspace_limits['z']
        
        if not (x_min <= x <= x_max):
            violations.append(f"X position {x:.3f} outside [{x_min:.3f}, {x_max:.3f}]")
        if not (y_min <= y <= y_max):
            violations.append(f"Y position {y:.3f} outside [{y_min:.3f}, {y_max:.3f}]") 
        if not (z_min <= z <= z_max):
            violations.append(f"Z position {z:.3f} outside [{z_min:.3f}, {z_max:.3f}]")
            
        if violations:
            return False, "; ".join(violations)
        return True, ""
    
    def check_joint_velocities(self, joint_velocities: np.ndarray) -> Tuple[bool, str]:
        """Check if joint velocities are within safe limits.
        
        Args:
            joint_velocities: Current joint velocities
            
        Returns:
            Tuple of (is_safe, warning_message)
        """
        violations = []
        for i, vel in enumerate(joint_velocities):
            if abs(vel) > self.max_joint_velocity:
                violations.append(f"Joint {i+1} velocity too high: {abs(vel):.3f} > {self.max_joint_velocity}")
        
        if violations:
            return False, "; ".join(violations)
        return True, ""
    
    def check_action_safety(self, action: np.ndarray, current_joints: np.ndarray, 
                           dt: float, action_scale: float) -> Tuple[np.ndarray, str]:
        """Check and modify action for safety.
        
        Args:
            action: Proposed action
            current_joints: Current joint positions
            dt: Control timestep  
            action_scale: Action scaling factor
            
        Returns:
            Tuple of (safe_action, warning_message)
        """
        warnings = []
        safe_action = action.copy()
        
        # Check action magnitude
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_action_norm:
            scale_factor = self.max_action_norm / action_norm
            safe_action = action * scale_factor
            warnings.append(f"Action norm reduced from {action_norm:.3f} to {self.max_action_norm}")
        
        # Check predicted joint positions
        predicted_joints = current_joints + (dt * safe_action * action_scale)
        is_safe, joint_msg = self.check_joint_limits(predicted_joints)
        
        if not is_safe:
            # Reduce action scale to stay within limits
            safe_action = safe_action * 0.5
            warnings.append(f"Action scaled down due to joint limits: {joint_msg}")
        
        warning_msg = "; ".join(warnings) if warnings else ""
        return safe_action, warning_msg
    
    def update_state(self, joint_positions: np.ndarray, ee_position: np.ndarray):
        """Update internal state for velocity tracking.
        
        Args:
            joint_positions: Current joint positions
            ee_position: Current end effector position  
        """
        current_time = time.time()
        
        if self._prev_joint_pos is not None and self._prev_time is not None:
            dt = current_time - self._prev_time
            if dt > 0:
                # Estimate velocities
                joint_vel = (joint_positions - self._prev_joint_pos) / dt
                ee_vel = np.linalg.norm((ee_position - self._prev_ee_pos) / dt)
                
                # Check velocity limits
                is_joint_safe, joint_msg = self.check_joint_velocities(joint_vel)
                if not is_joint_safe:
                    print(f"Warning: {joint_msg}")
                
                if ee_vel > self.max_ee_velocity:
                    print(f"Warning: End effector velocity too high: {ee_vel:.3f} > {self.max_ee_velocity}")
        
        self._prev_joint_pos = joint_positions.copy()
        self._prev_ee_pos = ee_position.copy()
        self._prev_time = current_time
    
    def set_workspace_limits(self, workspace_limits: dict):
        """Update workspace limits.
        
        Args:
            workspace_limits: Dictionary with 'x', 'y', 'z' limit tuples
        """
        self.workspace_limits.update(workspace_limits)
    
    def emergency_check(self, joint_positions: np.ndarray, ee_position: np.ndarray) -> Tuple[bool, str]:
        """Perform comprehensive emergency safety check.
        
        Args:
            joint_positions: Current joint positions
            ee_position: Current end effector position
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check joint limits with minimal buffer
        is_joint_safe, joint_msg = self.check_joint_limits(joint_positions, buffer=0.05)
        if not is_joint_safe:
            return False, f"EMERGENCY: Joint limits violated - {joint_msg}"
        
        # Check workspace limits
        is_workspace_safe, workspace_msg = self.check_workspace_limits(ee_position)
        if not is_workspace_safe:
            return False, f"EMERGENCY: Workspace violated - {workspace_msg}"
        
        return True, ""


# Simple emergency stop utility
class EmergencyStop:
    """Simple emergency stop utility."""
    
    def __init__(self):
        self.is_stopped = False
        self.stop_reason = ""
        self.stop_time = None
        
    def trigger(self, reason: str = "Manual emergency stop"):
        """Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        self.is_stopped = True
        self.stop_reason = reason
        self.stop_time = time.time()
        print(f"EMERGENCY STOP: {reason}")
        
    def reset(self) -> bool:
        """Reset emergency stop if safe to do so.
        
        Returns:
            True if reset successful, False otherwise
        """
        if not self.is_stopped:
            return True
            
        # Simple time-based reset (could add more sophisticated checks)
        if self.stop_time and (time.time() - self.stop_time) > 5.0:  # 5 second minimum
            self.is_stopped = False
            self.stop_reason = ""
            self.stop_time = None
            print("Emergency stop reset")
            return True
        else:
            print("Emergency stop reset denied - wait at least 5 seconds")
            return False
    
    def __bool__(self):
        """Return True if emergency stop is active."""
        return self.is_stopped


# Utility functions for common safety operations

def clip_action_to_safe_range(action: np.ndarray, max_norm: float = 0.5) -> np.ndarray:
    """Clip action to safe magnitude range.
    
    Args:
        action: Input action
        max_norm: Maximum allowed action norm
        
    Returns:
        Clipped action
    """
    action_norm = np.linalg.norm(action)
    if action_norm > max_norm:
        return action * (max_norm / action_norm)
    return action


def check_robot_ready(robot) -> Tuple[bool, str]:
    """Basic check if robot is ready for operation.
    
    Args:
        robot: Franky robot instance
        
    Returns:
        Tuple of (is_ready, status_message)
    """
    try:
        # Try to get robot state
        joint_state = robot.current_joint_state
        cartesian_state = robot.current_cartesian_state
        
        # Check if robot is moving
        if cartesian_state.is_moving:
            return False, "Robot is currently moving"
            
        # Check if we can read joint positions
        joint_positions = np.array(joint_state.position)
        if len(joint_positions) != 7:
            return False, f"Invalid joint state: got {len(joint_positions)} joints, expected 7"
            
        return True, "Robot ready"
        
    except Exception as e:
        return False, f"Robot communication error: {e}"


def safe_move_to_position(robot, target_joints: np.ndarray, safety_checker: FrankaSafetyChecker) -> bool:
    """Safely move robot to target joint positions.
    
    Args:
        robot: Franky robot instance
        target_joints: Target joint positions
        safety_checker: Safety checker instance
        
    Returns:
        True if move was successful, False otherwise
    """
    # Safety check on target position
    is_safe, msg = safety_checker.check_joint_limits(target_joints)
    if not is_safe:
        print(f"Target position unsafe: {msg}")
        return False
    
    try:
        from franky import JointMotion
        motion = JointMotion(target_joints)
        robot.move(motion)
        print("Move completed successfully")
        return True
        
    except Exception as e:
        print(f"Move failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Franka safety utilities...")
    
    # Test safety checker
    safety_checker = FrankaSafetyChecker()
    
    # Test joint limits
    safe_joints = np.array([0, -0.5, 0, -1.5, 0, 1.0, 0.5])
    unsafe_joints = np.array([3.0, -0.5, 0, -1.5, 0, 1.0, 0.5])  # First joint over limit
    
    is_safe, msg = safety_checker.check_joint_limits(safe_joints)
    print(f"Safe joints check: {is_safe} ({msg})")
    
    is_safe, msg = safety_checker.check_joint_limits(unsafe_joints)
    print(f"Unsafe joints check: {is_safe} ({msg})")
    
    # Test workspace limits
    safe_position = np.array([0.5, 0.0, 0.3])
    unsafe_position = np.array([1.0, 0.0, 0.3])  # Outside workspace
    
    is_safe, msg = safety_checker.check_workspace_limits(safe_position)
    print(f"Safe position check: {is_safe} ({msg})")
    
    is_safe, msg = safety_checker.check_workspace_limits(unsafe_position)
    print(f"Unsafe position check: {is_safe} ({msg})")
    
    # Test action safety
    safe_action = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
    unsafe_action = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])  # Too large
    
    modified_action, msg = safety_checker.check_action_safety(
        safe_action, safe_joints, dt=0.033, action_scale=0.1
    )
    print(f"Safe action: norm={np.linalg.norm(modified_action):.3f} ({msg})")
    
    modified_action, msg = safety_checker.check_action_safety(
        unsafe_action, safe_joints, dt=0.033, action_scale=0.1  
    )
    print(f"Unsafe action: norm={np.linalg.norm(modified_action):.3f} ({msg})")
    
    # Test emergency stop
    e_stop = EmergencyStop()
    print(f"Emergency stop active: {bool(e_stop)}")
    
    e_stop.trigger("Test stop")
    print(f"Emergency stop active: {bool(e_stop)}")
    
    print("\n✓ Safety utilities test completed")