#!/usr/bin/env python3
"""
ROS2 State Publisher with timer_callback and RPC functionality.
This script monitors joint states and periodically updates them via gRPC calls.
"""

import time
import json
import subprocess
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from typing import Optional


class ROS2StatePublisher(Node):
    """
    ROS2 State Publisher class that monitors joint states and updates them via gRPC.
    Includes timer_callback functionality for periodic monitoring and RPC calls.
    """
    
    def __init__(self):
        super().__init__('ros2_state_publisher')
        
        # Initialize parameters
        self.declare_parameter('timer_frequency', 10.0)  # Hz
        self.declare_parameter('log_interval_seconds', 5.0)  # How often to log states
        self.declare_parameter('grpc_server_endpoint', 'localhost:5001')  # gRPC server endpoint
        
        # Get parameters
        self.timer_frequency = self.get_parameter('timer_frequency').value
        self.log_interval_seconds = self.get_parameter('log_interval_seconds').value
        self.grpc_server_endpoint = self.get_parameter('grpc_server_endpoint').value
        
        # Initialize joint state storage
        self.latest_arm_joint_state = None
        self.latest_hand_joint_state = None
        self.latest_neck_joint_state = None
        
        # Setup QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Setup subscribers
        self.arm_joint_sub = self.create_subscription(
            JointState,
            '/motion/control/arm_joint_state',
            self.arm_joint_callback,
            qos
        )
        self.hand_joint_sub = self.create_subscription(
            JointState,
            '/motion/control/hand_joint_state',
            self.hand_joint_callback,
            qos
        )
        self.neck_joint_sub = self.create_subscription(
            JointState,
            '/motion/control/neck_joint_state',
            self.neck_joint_callback,
            qos
        )
        
        # Setup timer for periodic tasks
        timer_period = 1.0 / self.timer_frequency  # Convert Hz to seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Initialize timer counter for periodic logging and gRPC updates
        self._timer_count = 0
        self._log_interval_count = int(self.log_interval_seconds * self.timer_frequency)
        
        self.get_logger().info(f'ROS2 State Publisher initialized - Timer frequency: {self.timer_frequency} Hz')
        self.get_logger().info(f'Log interval: {self.log_interval_seconds} seconds')
        self.get_logger().info(f'gRPC updates: every timer tick')
        self.get_logger().info(f'gRPC server endpoint: {self.grpc_server_endpoint}')
    
    def arm_joint_callback(self, msg: JointState):
        """Callback for arm joint state subscription."""
        self.latest_arm_joint_state = msg
        self.get_logger().debug(f'Received arm joint state: {len(msg.name)} joints')
    
    def hand_joint_callback(self, msg: JointState):
        """Callback for hand joint state subscription."""
        self.latest_hand_joint_state = msg
        self.get_logger().debug(f'Received hand joint state: {len(msg.name)} joints')
    
    def neck_joint_callback(self, msg: JointState):
        """Callback for neck joint state subscription."""
        self.latest_neck_joint_state = msg
        self.get_logger().debug(f'Received neck joint state: {len(msg.name)} joints')
    
    def timer_callback(self):
        """Periodic timer callback for monitoring and maintenance."""
        # Increment timer counter
        self._timer_count += 1
        
        # Log connection status periodically
        self.get_logger().debug('ROS2 State Publisher running - monitoring joint states')
        
        # Log latest states every specified interval
        if self._timer_count % self._log_interval_count == 0:
            self.log_latest_states()
        
        # Update gRPC server on every timer tick
        self.update_grpc_joint_states()
    
    def get_latest_arm_state(self) -> Optional[JointState]:
        """Get the latest arm joint state."""
        return self.latest_arm_joint_state
    
    def get_latest_hand_state(self) -> Optional[JointState]:
        """Get the latest hand joint state."""
        return self.latest_hand_joint_state
    
    def get_latest_neck_state(self) -> Optional[JointState]:
        """Get the latest neck joint state."""
        return self.latest_neck_joint_state
    
    def update_grpc_joint_states(self):
        """Update joint states on the gRPC server using grpcurl."""
        try:
            # Collect all joint states
            joint_updates = {}
            
            # Add arm joint states
            if self.latest_arm_joint_state:
                for i, joint_name in enumerate(self.latest_arm_joint_state.name):
                    if i < len(self.latest_arm_joint_state.position):
                        joint_updates[joint_name] = self.latest_arm_joint_state.position[i]
            
            # Add hand joint states
            if self.latest_hand_joint_state:
                for i, joint_name in enumerate(self.latest_hand_joint_state.name):
                    if i < len(self.latest_hand_joint_state.position):
                        joint_updates[joint_name] = self.latest_hand_joint_state.position[i]
            
            # Add neck joint states
            if self.latest_neck_joint_state:
                for i, joint_name in enumerate(self.latest_neck_joint_state.name):
                    if i < len(self.latest_neck_joint_state.position):
                        joint_updates[joint_name] = self.latest_neck_joint_state.position[i]
            
            if not joint_updates:
                self.get_logger().warn('No joint states available for gRPC update')
                return
            
            # Create JSON payload
            payload = {
                "joint_updates": joint_updates
            }
            
            # Convert to JSON string
            json_payload = json.dumps(payload)
            
            # Build grpcurl command
            cmd = [
                'grpcurl', '-plaintext', '-d', json_payload,
                self.grpc_server_endpoint, 'rosbot_api.RobotApiService/UpdateJoints'
            ]
            
            # Execute the command
            self.get_logger().info(f'Updating {len(joint_updates)} joints on gRPC server: {self.grpc_server_endpoint}')
            self.get_logger().debug(f'gRPC command: {" ".join(cmd)}')
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
            
            if result.returncode == 0:
                self.get_logger().info(f'Successfully updated joints on gRPC server. Response: {result.stdout.strip()}')
            else:
                self.get_logger().error(f'Failed to update joints on gRPC server. Error: {result.stderr.strip()}')
                
        except subprocess.TimeoutExpired:
            self.get_logger().error('gRPC update timed out')
        except FileNotFoundError:
            self.get_logger().error('grpcurl not found. Please install grpcurl.')
        except Exception as e:
            self.get_logger().error(f'Error updating gRPC joint states: {e}')
    
    def log_latest_states(self):
        """Log the latest received joint states."""
        self.get_logger().info("=" * 50)
        self.get_logger().info("PERIODIC STATUS REPORT")
        self.get_logger().info("=" * 50)
        
        if self.latest_arm_joint_state:
            arm_names = self.latest_arm_joint_state.name
            arm_positions = self.latest_arm_joint_state.position
            self.get_logger().info(f'Latest ARM STATE - {len(arm_names)} joints')
            self.get_logger().info(f'  Joint names: {arm_names[:3]}...' + (f' (and {len(arm_names)-3} more)' if len(arm_names) > 3 else ''))
            if arm_positions:
                self.get_logger().info(f'  Sample positions: {arm_positions[:3]}...')
        else:
            self.get_logger().info('Latest ARM STATE - No data received yet')
            
        if self.latest_hand_joint_state:
            hand_names = self.latest_hand_joint_state.name
            hand_positions = self.latest_hand_joint_state.position
            self.get_logger().info(f'Latest HAND STATE - {len(hand_names)} joints')
            self.get_logger().info(f'  Joint names: {hand_names[:3]}...' + (f' (and {len(hand_names)-3} more)' if len(hand_names) > 3 else ''))
            if hand_positions:
                self.get_logger().info(f'  Sample positions: {hand_positions[:3]}...')
        else:
            self.get_logger().info('Latest HAND STATE - No data received yet')
            
        if self.latest_neck_joint_state:
            neck_names = self.latest_neck_joint_state.name
            neck_positions = self.latest_neck_joint_state.position
            self.get_logger().info(f'Latest NECK STATE - {len(neck_names)} joints')
            self.get_logger().info(f'  Joint names: {neck_names[:3]}...' + (f' (and {len(neck_names)-3} more)' if len(neck_names) > 3 else ''))
            if neck_positions:
                self.get_logger().info(f'  Sample positions: {neck_positions[:3]}...')
        else:
            self.get_logger().info('Latest NECK STATE - No data received yet')
        
        # Log timer statistics
        uptime_seconds = self._timer_count / self.timer_frequency
        self.get_logger().info(f'Uptime: {uptime_seconds:.1f} seconds')
        self.get_logger().info(f'Timer calls: {self._timer_count}')
        self.get_logger().info("=" * 50)
    
    def shutdown(self):
        """Clean shutdown of the node."""
        self.get_logger().info('ROS2 State Publisher shutdown complete')


def main(args=None):
    rclpy.init(args=args)
    
    publisher = ROS2StatePublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.shutdown()
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()