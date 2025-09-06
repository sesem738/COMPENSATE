#!/usr/bin/env python3

"""
Safety monitoring system for real-world Franka robot environment.

This module provides comprehensive safety monitoring, emergency handling,
and recovery capabilities for safe real-world robot operation.
"""

from __future__ import annotations

import logging
import numpy as np
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# Try to import franky for real robot integration
try:
    from franky import Robot, Gripper, Errors
    FRANKY_AVAILABLE = True
except ImportError:
    print("Warning: franky library not available")
    Robot = Gripper = Errors = None
    FRANKY_AVAILABLE = False


class SafetyLevel(Enum):
    """Safety alert levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyViolationType(Enum):
    """Types of safety violations."""
    JOINT_LIMIT = "joint_limit"
    WORKSPACE_LIMIT = "workspace_limit"
    VELOCITY_LIMIT = "velocity_limit"
    ACCELERATION_LIMIT = "acceleration_limit"
    FORCE_LIMIT = "force_limit"
    COMMUNICATION_ERROR = "communication_error"
    ROBOT_ERROR = "robot_error"
    USER_EMERGENCY_STOP = "user_emergency_stop"


@dataclass
class SafetyEvent:
    """Container for safety event information."""
    
    timestamp: float
    level: SafetyLevel
    violation_type: SafetyViolationType
    message: str
    robot_state: Optional[Dict] = None
    action_taken: Optional[str] = None
    resolved: bool = False


@dataclass
class SafetyLimits:
    """Container for safety limit parameters."""
    
    # Joint limits (radians)
    joint_position_min: np.ndarray = field(default_factory=lambda: np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]))
    joint_position_max: np.ndarray = field(default_factory=lambda: np.array(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]))
    joint_position_buffer: float = 0.1
    
    # Velocity limits (rad/s)
    joint_velocity_max: np.ndarray = field(default_factory=lambda: np.array(
        [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]))
    end_effector_velocity_max: float = 1.0  # m/s
    
    # Acceleration limits (rad/s²)
    joint_acceleration_max: np.ndarray = field(default_factory=lambda: np.array(
        [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]))
    end_effector_acceleration_max: float = 5.0  # m/s²
    
    # Force limits (N)
    contact_force_max: float = 20.0
    
    # Workspace limits [x_min, x_max, y_min, y_max, z_min, z_max]
    workspace_limits: np.ndarray = field(default_factory=lambda: np.array(
        [0.2, 0.8, -0.4, 0.4, 0.05, 0.8]))
    
    # Communication limits
    communication_timeout: float = 1.0  # seconds
    max_consecutive_errors: int = 3


class SafetyMonitor:
    """
    Comprehensive safety monitoring system for real-world robot operation.
    
    Features:
    - Real-time safety limit monitoring
    - Emergency stop functionality
    - Automatic recovery procedures
    - Safety event logging and reporting
    - Multi-threaded monitoring for responsiveness
    """
    
    def __init__(self, robot: Optional[Robot] = None, 
                 gripper: Optional[Gripper] = None,
                 limits: Optional[SafetyLimits] = None,
                 log_level: int = logging.INFO):
        """Initialize safety monitor.
        
        Args:
            robot: Franky robot instance
            gripper: Franky gripper instance  
            limits: Safety limits configuration
            log_level: Logging level
        """
        self.robot = robot
        self.gripper = gripper
        self.limits = limits or SafetyLimits()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Safety state
        self.is_monitoring = False
        self.emergency_stop_active = False
        self.safety_events: List[SafetyEvent] = []
        self.current_safety_level = SafetyLevel.NORMAL
        
        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # State tracking for derivative calculations
        self._previous_joint_positions: Optional[np.ndarray] = None
        self._previous_joint_velocities: Optional[np.ndarray] = None
        self._previous_ee_position: Optional[np.ndarray] = None
        self._previous_ee_velocity: Optional[np.ndarray] = None
        self._previous_timestamp = 0.0
        
        # Error tracking
        self._consecutive_comm_errors = 0
        self._last_successful_update = time.time()
        
        # Safety callbacks
        self._safety_callbacks: Dict[SafetyLevel, List[Callable]] = {
            level: [] for level in SafetyLevel
        }
        
        self.logger.info("Safety monitor initialized")
        
    def register_callback(self, level: SafetyLevel, callback: Callable[[SafetyEvent], None]):
        """Register callback for safety events of specified level.
        
        Args:
            level: Safety level to trigger callback
            callback: Function to call when event occurs
        """
        self._safety_callbacks[level].append(callback)
        self.logger.debug(f"Registered callback for {level.value} events")
        
    def start_monitoring(self, frequency: float = 50.0):
        """Start safety monitoring in background thread.
        
        Args:
            frequency: Monitoring frequency in Hz
        """
        if self.is_monitoring:
            self.logger.warning("Safety monitoring already active")
            return
            
        if not FRANKY_AVAILABLE or self.robot is None:
            self.logger.error("Cannot start monitoring: robot not available")
            return
            
        self.is_monitoring = True
        self._stop_monitoring.clear()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(frequency,),
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info(f"Safety monitoring started at {frequency} Hz")
        
    def stop_monitoring(self):
        """Stop safety monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        self._stop_monitoring.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
            
        self.logger.info("Safety monitoring stopped")
        
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        self.emergency_stop_active = True
        
        # Log emergency event
        event = SafetyEvent(
            timestamp=time.time(),
            level=SafetyLevel.EMERGENCY,
            violation_type=SafetyViolationType.USER_EMERGENCY_STOP,
            message=reason,
            action_taken="emergency_stop"
        )
        self._log_safety_event(event)
        
        # Stop robot if available
        if self.robot is not None:
            try:
                self.robot.stop()
                self.logger.critical(f"Emergency stop activated: {reason}")
            except Exception as e:
                self.logger.error(f"Failed to stop robot during emergency: {e}")
        else:
            self.logger.critical(f"Emergency stop requested but no robot available: {reason}")
            
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop if conditions are safe.
        
        Returns:
            True if successfully reset, False otherwise
        """
        if not self.emergency_stop_active:
            return True
            
        # Check if it's safe to reset
        if not self._check_safe_to_reset():
            self.logger.warning("Cannot reset emergency stop: conditions not safe")
            return False
            
        self.emergency_stop_active = False
        self.current_safety_level = SafetyLevel.NORMAL
        
        self.logger.info("Emergency stop reset - normal operation resumed")
        return True
        
    def check_action_safety(self, action: np.ndarray, 
                           current_state: Dict) -> Tuple[bool, Optional[str]]:
        """Check if proposed action is safe to execute.
        
        Args:
            action: Proposed action to check
            current_state: Current robot state
            
        Returns:
            Tuple of (is_safe, reason_if_unsafe)
        """
        if self.emergency_stop_active:
            return False, "Emergency stop active"
            
        try:
            # Predict resulting joint positions
            current_joints = np.array(current_state.get('joint_positions', np.zeros(7)))
            predicted_joints = current_joints + action
            
            # Check joint limits
            safety_min = self.limits.joint_position_min + self.limits.joint_position_buffer
            safety_max = self.limits.joint_position_max - self.limits.joint_position_buffer
            
            if np.any(predicted_joints < safety_min) or np.any(predicted_joints > safety_max):
                return False, "Action would violate joint limits"
                
            # Check action magnitude (rate limiting)
            if np.linalg.norm(action) > 0.2:  # Conservative action limit
                return False, "Action magnitude too large"
                
            # Check workspace limits (if end effector position available)
            if 'end_effector_position' in current_state:
                ee_pos = np.array(current_state['end_effector_position'])
                if not self._check_workspace_limits(ee_pos):
                    return False, "End effector outside workspace"
                    
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error checking action safety: {e}")
            return False, f"Safety check failed: {e}"
            
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report.
        
        Returns:
            Dictionary containing safety status and statistics
        """
        now = time.time()
        recent_events = [e for e in self.safety_events if now - e.timestamp < 3600]  # Last hour
        
        report = {
            "monitoring_active": self.is_monitoring,
            "emergency_stop_active": self.emergency_stop_active,
            "current_safety_level": self.current_safety_level.value,
            "total_safety_events": len(self.safety_events),
            "recent_safety_events": len(recent_events),
            "consecutive_comm_errors": self._consecutive_comm_errors,
            "last_update": self._last_successful_update,
            "time_since_last_update": now - self._last_successful_update,
        }
        
        # Event breakdown by type
        event_counts = {}
        for event in recent_events:
            event_type = event.violation_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        report["recent_event_breakdown"] = event_counts
        
        # Safety limits status
        report["safety_limits"] = {
            "joint_position_buffer": self.limits.joint_position_buffer,
            "max_joint_velocity": self.limits.joint_velocity_max.tolist(),
            "max_ee_velocity": self.limits.end_effector_velocity_max,
            "max_contact_force": self.limits.contact_force_max,
            "workspace_limits": self.limits.workspace_limits.tolist(),
        }
        
        return report
        
    def _monitor_loop(self, frequency: float):
        """Main monitoring loop (runs in background thread).
        
        Args:
            frequency: Monitoring frequency in Hz
        """
        dt = 1.0 / frequency
        
        self.logger.debug(f"Starting safety monitoring loop at {frequency} Hz")
        
        while not self._stop_monitoring.is_set():
            start_time = time.time()
            
            try:
                self._perform_safety_checks()
            except Exception as e:
                self.logger.error(f"Error in safety monitoring loop: {e}")
                self._handle_monitoring_error(e)
            
            # Sleep for remaining time to maintain frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            
            if self._stop_monitoring.wait(timeout=sleep_time):
                break  # Stop requested
                
        self.logger.debug("Safety monitoring loop stopped")
        
    def _perform_safety_checks(self):
        """Perform all safety checks on current robot state."""
        if not self.robot:
            return
            
        try:
            # Get current robot state
            joint_state = self.robot.current_joint_state
            cartesian_state = self.robot.current_cartesian_state
            
            current_time = time.time()
            joint_positions = np.array(joint_state.position)
            joint_velocities = np.array(joint_state.velocity)
            
            ee_pose = cartesian_state.pose.end_effector_pose
            ee_position = np.array([
                ee_pose.translation[0],
                ee_pose.translation[1],
                ee_pose.translation[2]
            ])
            
            # Update successful communication
            self._last_successful_update = current_time
            self._consecutive_comm_errors = 0
            
            # Perform individual safety checks
            self._check_joint_limits(joint_positions)
            self._check_joint_velocities(joint_velocities)
            self._check_workspace_limits(ee_position)
            
            # Check accelerations if we have previous data
            if self._previous_joint_positions is not None:
                dt = current_time - self._previous_timestamp
                if dt > 0:
                    self._check_joint_accelerations(joint_positions, joint_velocities, dt)
                    self._check_ee_acceleration(ee_position, dt)
            
            # Update previous state for next iteration
            self._previous_joint_positions = joint_positions.copy()
            self._previous_joint_velocities = joint_velocities.copy()
            self._previous_ee_position = ee_position.copy()
            self._previous_timestamp = current_time
            
        except Exception as e:
            self._consecutive_comm_errors += 1
            
            if self._consecutive_comm_errors >= self.limits.max_consecutive_errors:
                self._trigger_safety_event(
                    SafetyLevel.CRITICAL,
                    SafetyViolationType.COMMUNICATION_ERROR,
                    f"Communication failed {self._consecutive_comm_errors} times: {e}"
                )
            else:
                self.logger.warning(f"Communication error ({self._consecutive_comm_errors}/{self.limits.max_consecutive_errors}): {e}")
                
    def _check_joint_limits(self, joint_positions: np.ndarray):
        """Check joint position limits."""
        safety_min = self.limits.joint_position_min + self.limits.joint_position_buffer
        safety_max = self.limits.joint_position_max - self.limits.joint_position_buffer
        
        violations = []
        for i, (pos, min_pos, max_pos) in enumerate(zip(joint_positions, safety_min, safety_max)):
            if pos < min_pos or pos > max_pos:
                violations.append(f"Joint {i+1}: {pos:.3f} (limits: {min_pos:.3f}, {max_pos:.3f})")
        
        if violations:
            self._trigger_safety_event(
                SafetyLevel.CRITICAL,
                SafetyViolationType.JOINT_LIMIT,
                f"Joint limit violations: {'; '.join(violations)}"
            )
            
    def _check_joint_velocities(self, joint_velocities: np.ndarray):
        """Check joint velocity limits."""
        violations = []
        for i, (vel, max_vel) in enumerate(zip(np.abs(joint_velocities), self.limits.joint_velocity_max)):
            if vel > max_vel:
                violations.append(f"Joint {i+1}: {vel:.3f} rad/s (limit: {max_vel:.3f})")
        
        if violations:
            level = SafetyLevel.CRITICAL if len(violations) > 2 else SafetyLevel.WARNING
            self._trigger_safety_event(
                level,
                SafetyViolationType.VELOCITY_LIMIT,
                f"Joint velocity violations: {'; '.join(violations)}"
            )
            
    def _check_workspace_limits(self, ee_position: np.ndarray) -> bool:
        """Check workspace limits.
        
        Args:
            ee_position: End effector position [x, y, z]
            
        Returns:
            True if within limits, False otherwise
        """
        x, y, z = ee_position
        x_min, x_max, y_min, y_max, z_min, z_max = self.limits.workspace_limits
        
        violations = []
        if x < x_min or x > x_max:
            violations.append(f"X: {x:.3f} (limits: {x_min:.3f}, {x_max:.3f})")
        if y < y_min or y > y_max:
            violations.append(f"Y: {y:.3f} (limits: {y_min:.3f}, {y_max:.3f})")
        if z < z_min or z > z_max:
            violations.append(f"Z: {z:.3f} (limits: {z_min:.3f}, {z_max:.3f})")
        
        if violations:
            self._trigger_safety_event(
                SafetyLevel.CRITICAL,
                SafetyViolationType.WORKSPACE_LIMIT,
                f"Workspace violations: {'; '.join(violations)}"
            )
            return False
            
        return True
        
    def _check_joint_accelerations(self, joint_positions: np.ndarray, 
                                  joint_velocities: np.ndarray, dt: float):
        """Check joint acceleration limits."""
        if self._previous_joint_velocities is None:
            return
            
        joint_accelerations = (joint_velocities - self._previous_joint_velocities) / dt
        
        violations = []
        for i, (acc, max_acc) in enumerate(zip(np.abs(joint_accelerations), self.limits.joint_acceleration_max)):
            if acc > max_acc:
                violations.append(f"Joint {i+1}: {acc:.2f} rad/s² (limit: {max_acc:.2f})")
        
        if violations:
            self._trigger_safety_event(
                SafetyLevel.WARNING,
                SafetyViolationType.ACCELERATION_LIMIT,
                f"Joint acceleration violations: {'; '.join(violations)}"
            )
            
    def _check_ee_acceleration(self, ee_position: np.ndarray, dt: float):
        """Check end effector acceleration limits."""
        if self._previous_ee_position is None or self._previous_ee_velocity is None:
            return
            
        ee_velocity = (ee_position - self._previous_ee_position) / dt
        ee_acceleration = np.linalg.norm((ee_velocity - self._previous_ee_velocity) / dt)
        
        if ee_acceleration > self.limits.end_effector_acceleration_max:
            self._trigger_safety_event(
                SafetyLevel.WARNING,
                SafetyViolationType.ACCELERATION_LIMIT,
                f"End effector acceleration: {ee_acceleration:.2f} m/s² (limit: {self.limits.end_effector_acceleration_max:.2f})"
            )
            
        self._previous_ee_velocity = ee_velocity.copy()
        
    def _trigger_safety_event(self, level: SafetyLevel, 
                             violation_type: SafetyViolationType, 
                             message: str,
                             robot_state: Optional[Dict] = None):
        """Trigger safety event and take appropriate action.
        
        Args:
            level: Safety level of the event
            violation_type: Type of safety violation
            message: Descriptive message
            robot_state: Current robot state (if available)
        """
        event = SafetyEvent(
            timestamp=time.time(),
            level=level,
            violation_type=violation_type,
            message=message,
            robot_state=robot_state
        )
        
        # Take action based on safety level
        if level == SafetyLevel.EMERGENCY or level == SafetyLevel.CRITICAL:
            self.emergency_stop(f"Safety violation: {message}")
            event.action_taken = "emergency_stop"
        elif level == SafetyLevel.WARNING:
            self.logger.warning(f"Safety warning: {message}")
            event.action_taken = "warning_logged"
        
        # Update current safety level
        if level.value in ['emergency', 'critical'] or (
            level == SafetyLevel.WARNING and self.current_safety_level == SafetyLevel.NORMAL):
            self.current_safety_level = level
            
        # Log and store event
        self._log_safety_event(event)
        
        # Execute callbacks
        for callback in self._safety_callbacks[level]:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in safety callback: {e}")
                
    def _log_safety_event(self, event: SafetyEvent):
        """Log safety event."""
        self.safety_events.append(event)
        
        log_msg = f"Safety {event.level.value}: {event.message}"
        if event.action_taken:
            log_msg += f" (Action: {event.action_taken})"
            
        if event.level == SafetyLevel.EMERGENCY:
            self.logger.critical(log_msg)
        elif event.level == SafetyLevel.CRITICAL:
            self.logger.error(log_msg)
        elif event.level == SafetyLevel.WARNING:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
            
        # Limit event history size
        if len(self.safety_events) > 1000:
            self.safety_events = self.safety_events[-500:]  # Keep recent half
            
    def _handle_monitoring_error(self, error: Exception):
        """Handle errors in monitoring loop."""
        self.logger.error(f"Safety monitoring error: {error}")
        
        # If it's a critical error, trigger emergency stop
        if "connection" in str(error).lower() or "timeout" in str(error).lower():
            self._trigger_safety_event(
                SafetyLevel.CRITICAL,
                SafetyViolationType.COMMUNICATION_ERROR,
                f"Monitoring error: {error}"
            )
            
    def _check_safe_to_reset(self) -> bool:
        """Check if conditions are safe to reset emergency stop.
        
        Returns:
            True if safe to reset, False otherwise
        """
        # Check recent safety events
        recent_critical_events = [
            e for e in self.safety_events
            if (time.time() - e.timestamp < 30.0 and  # Last 30 seconds
                e.level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY] and
                not e.resolved)
        ]
        
        if recent_critical_events:
            self.logger.warning(f"Cannot reset: {len(recent_critical_events)} unresolved critical events")
            return False
            
        # Check communication status
        if time.time() - self._last_successful_update > self.limits.communication_timeout:
            self.logger.warning("Cannot reset: communication timeout")
            return False
            
        return True
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Example usage and testing
if __name__ == "__main__":
    print("Testing safety monitoring system...")
    
    # Test with mock robot (no physical connection)
    safety_monitor = SafetyMonitor()
    
    # Test safety limit checks
    limits = SafetyLimits()
    
    # Test joint limit check
    safe_joints = np.array([0, -0.5, 0, -1.5, 0, 1.0, 0.5])
    unsafe_joints = np.array([3.0, -0.5, 0, -1.5, 0, 1.0, 0.5])  # First joint over limit
    
    print("Testing joint limits:")
    print(f"Safe joints: {safe_joints}")
    print(f"Unsafe joints: {unsafe_joints}")
    
    # Test workspace check
    safe_position = np.array([0.5, 0.0, 0.3])
    unsafe_position = np.array([1.0, 0.0, 0.3])  # Outside workspace
    
    print(f"\nTesting workspace limits:")
    print(f"Safe position: {safe_position}")
    print(f"Unsafe position: {unsafe_position}")
    
    # Test action safety check
    mock_state = {
        'joint_positions': safe_joints,
        'end_effector_position': safe_position
    }
    
    safe_action = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01])
    unsafe_action = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    
    is_safe, reason = safety_monitor.check_action_safety(safe_action, mock_state)
    print(f"\nSafe action check: {is_safe} (reason: {reason})")
    
    is_safe, reason = safety_monitor.check_action_safety(unsafe_action, mock_state)  
    print(f"Unsafe action check: {is_safe} (reason: {reason})")
    
    # Test safety report
    report = safety_monitor.get_safety_report()
    print(f"\nSafety report: {report}")
    
    print("\n✓ Safety monitoring system test completed")