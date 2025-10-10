#!/usr/bin/env python3
#  grpcurl -format json -plaintext -d '{"update_frequency": 100.0}' localhost:5000 rosbot_api.RobotApiService/StreamJointData | python3 ros2_streamer.py 

import time
import json
import threading
import sys
import subprocess
from collections import deque
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState


class ROS2Streamer(Node):
    """
    ROS2 Streamer class that subscribes to joint states and publishes joint commands
    based on JSON input from grpcurl stream piped to stdin.
    """
    
    def __init__(self):
        super().__init__('ros2_streamer')
        
        # Initialize parameters
        self.declare_parameter('update_frequency', 100.0)
        self.declare_parameter('max_arm_joint_change_rad', 0.2)  # Maximum change in radians per update for arm joints
        self.declare_parameter('max_hand_joint_change_rad', 200)  # Maximum change in radians per update for hand joints
        self.declare_parameter('max_neck_joint_change_rad', 0.2)  # Maximum change in radians per update for neck joints
        self.declare_parameter('grpc_server_endpoint', 'localhost:5001')  # gRPC server endpoint for joint updates
        self.declare_parameter('arm_joint_names', 
            [
                'idx13_left_arm_joint1', 
                'idx14_left_arm_joint2', 
                'idx15_left_arm_joint3', 
                'idx16_left_arm_joint4', 
                'idx17_left_arm_joint5', 
                'idx18_left_arm_joint6', 
                'idx19_left_arm_joint7', 
                'idx20_right_arm_joint1', 
                'idx21_right_arm_joint2', 
                'idx22_right_arm_joint3', 
                'idx23_right_arm_joint4', 
                'idx24_right_arm_joint5', 
                'idx25_right_arm_joint6', 
                'idx26_right_arm_joint7', 
            ]
        )
        self.declare_parameter('hand_joint_names', 
            [
                'left_thumb_0',
                'left_thumb_1',
                'left_index',
                'left_middle',
                'left_ring',
                'left_pinky',
                'right_thumb_0',
                'right_thumb_1',
                'right_index',
                'right_middle',
                'right_ring',
                'right_pinky',
            ]
        )
        self.declare_parameter('neck_joint_names', 
            [
                'idx27_head_joint1',
                'idx28_head_joint2',
            ]
        )
        
        # Get parameters
        self.update_frequency = self.get_parameter('update_frequency').value
        self.max_arm_joint_change_rad = self.get_parameter('max_arm_joint_change_rad').value
        self.max_hand_joint_change_rad = self.get_parameter('max_hand_joint_change_rad').value
        self.max_neck_joint_change_rad = self.get_parameter('max_neck_joint_change_rad').value
        self.grpc_server_endpoint = self.get_parameter('grpc_server_endpoint').value
        self.arm_joint_names = self.get_parameter('arm_joint_names').value
        self.hand_joint_names = self.get_parameter('hand_joint_names').value
        self.neck_joint_names = self.get_parameter('neck_joint_names').value
        
        # Initialize joint state storage
        self.arm_joint_state = JointState()
        self.hand_joint_state = JointState()
        self.neck_joint_state = JointState()
        self.latest_arm_joint_state = None
        self.latest_hand_joint_state = None
        self.latest_neck_joint_state = None
        
        # Initialize previous command states for change limiting
        # (Will use current joint states from subscriptions instead)
        
        # Initialize JSON history storage
        self.json_history = []
        self.max_history_size = 10  # Keep last 10 entries
        
        # Threading
        self.stdin_thread = None
        self.running = False
        
        # Setup QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Setup publishers
        self.arm_joint_pub = self.create_publisher(
            JointState, 
            '/motion/control/arm_joint_command', 
            10
        )
        self.hand_joint_pub = self.create_publisher(
            JointState, 
            '/motion/control/hand_joint_command', 
            10
        )
        self.neck_joint_pub = self.create_publisher(
            JointState, 
            '/motion/control/neck_joint_command', 
            10
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
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz timer
        
        # Start stdin reading thread
        self.start_stdin_reading()
        
        self.get_logger().info('ROS2 Streamer initialized - reading from stdin')
        self.get_logger().info('Expected grpcurl command:')
        self.get_logger().info('grpcurl -format json -plaintext -d \'{"update_frequency": 100.0}\' localhost:5000 rosbot_api.RobotApiService/StreamJointData | python3 ros2_streamer.py')
    
    def save_to_history(self, data: Dict):
        """Save JSON data to history with timestamp."""
        import time
        history_entry = {
            'timestamp': time.time(),
            'data': data
        }
        
        self.json_history.append(history_entry)
        
        # Maintain history size limit
        if len(self.json_history) > self.max_history_size:
            self.json_history.pop(0)  # Remove oldest entry
        
        self.get_logger().info(f'History size: {len(self.json_history)}/{self.max_history_size}')
    
    def get_history(self, count: int = None) -> List[Dict]:
        """Get the JSON history. If count is specified, return the last 'count' entries."""
        if count is None:
            return self.json_history.copy()
        else:
            return self.json_history[-count:].copy()
    
    def clear_history(self):
        """Clear the JSON history."""
        self.json_history.clear()
        self.get_logger().info('JSON history cleared')
    
    def print_history_summary(self):
        """Print a summary of the JSON history."""
        if not self.json_history:
            print("No JSON history available")
            return
        
        print(f"\nJSON History Summary ({len(self.json_history)} entries):")
        print("=" * 50)
        
        for i, entry in enumerate(self.json_history[-5:]):  # Show last 5 entries
            timestamp = entry['timestamp']
            data = entry['data']
            
            # Extract joint count info
            joint_positions = data.get('jointPositions', {})
            custom_joint_positions = data.get('customJointPositions', {})
            total_joints = len(joint_positions) + len(custom_joint_positions)
            
            print(f"Entry {len(self.json_history) - 5 + i + 1}:")
            print(f"  Timestamp: {timestamp}")
            print(f"  Total joints: {total_joints}")
            print(f"  Regular joints: {len(joint_positions)}")
            print(f"  Custom joints: {len(custom_joint_positions)}")
            print(f"  Sample joints: {list(joint_positions.keys())[:3]}...")
            print()
    
    def start_stdin_reading(self):
        """Start reading from stdin in a separate thread."""
        self.running = True
        self.stdin_thread = threading.Thread(target=self.stdin_worker, daemon=True)
        self.stdin_thread.start()
        self.get_logger().info('Started stdin reading thread')
    
    def stdin_worker(self):
        """Worker thread for reading JSON data from stdin."""
        try:
            # Buffer to accumulate JSON data
            json_buffer = ""
            brace_count = 0
            in_string = False
            escape_next = False
            
            for line in sys.stdin:
                if not self.running:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Add line to buffer
                json_buffer += line
                
                # Count braces to detect complete JSON objects
                for char in line:
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                
                # If we have a complete JSON object (balanced braces)
                if brace_count == 0 and json_buffer.strip():
                    try:
                        # Parse the complete JSON object
                        data = json.loads(json_buffer.strip())
                        
                        # Print the ingested JSON
                        #print(f"INGESTED JSON: {json.dumps(data, indent=2)}")
                        
                        # Save to history
                        self.save_to_history(data)
                        
                        self.process_json_joint_state(data)
                        # Clear buffer after successful parsing
                        json_buffer = ""
                    except json.JSONDecodeError as e:
                        self.get_logger().warn(f'Failed to parse JSON from stdin: {e}')
                        self.get_logger().warn(f'Raw buffer: {json_buffer}')
                        # Clear buffer on error to avoid accumulating bad data
                        json_buffer = ""
                    except Exception as e:
                        self.get_logger().error(f'Error processing stdin data: {e}')
                        json_buffer = ""
                    
        except Exception as e:
            self.get_logger().error(f'Stdin reading error: {e}')
    
    def process_json_joint_state(self, data: Dict):
        """Process incoming JSON joint state and publish to ROS."""
        try:
            # Log the received data for debugging
            self.get_logger().info(f'Received JSON data: {data}')
            
            # Extract joint positions from the JSON structure
            # Based on the proto schema, we expect:
            # {
            #   "jointPositions": {"joint_name": value, ...},
            #   "customJointPositions": {"custom_joint_name": value, ...},
            #   "timestamp": {...}
            # }
            
            joint_positions = data.get('joint_positions', {})
            custom_joint_positions = data.get('custom_joint_positions', {})
            
            # Combine regular and custom joint positions
            all_joint_positions = {**joint_positions, **custom_joint_positions}
            
            
            # Process arm joints - use arm_joint_names from regular jointPositions
            arm_positions = []
            arm_names = []
            
            for joint_name in self.arm_joint_names:
                if joint_name in joint_positions:
                    # Ensure the value is a float
                    value = float(joint_positions[joint_name])
                    arm_positions.append(value)
                    arm_names.append(joint_name)
            
            # Process hand joints - use hand_joint_names from customJointPositions
            hand_positions = []
            hand_names = []
            
            for joint_name in self.hand_joint_names:
                if joint_name in custom_joint_positions:
                    # Set _middle, _ring, and _pinky to 0, use original value for others
                    if '_middle' in joint_name or '_ring' in joint_name or '_pinky' in joint_name:
                        value = 2000.0
                    
                    elif 'right_thumb_0' in joint_name or 'right_thumb_1' in joint_name:
                        value = float(custom_joint_positions["right_thumb_0"])
                    elif 'left_thumb_0' in joint_name or 'left_thumb_1' in joint_name:
                        value = float(custom_joint_positions["left_thumb_0"])

                    else:
                        # Ensure the value is a float
                        value = float(custom_joint_positions[joint_name])
                    hand_positions.append(value)
                    hand_names.append(joint_name)
            
            # Process neck joints - use neck_joint_names from customJointPositions
            neck_positions = []
            neck_names = []
            
            for joint_name in self.neck_joint_names:
                if joint_name in custom_joint_positions:
                    # Ensure the value is a float
                    value = float(custom_joint_positions[joint_name])
                    neck_positions.append(value)
                    neck_names.append(joint_name)
            
            # Create and publish arm joint command
            if arm_names:
                # Limit joint changes based on current joint state
                limited_arm_positions = self.limit_joint_changes(
                    arm_names, arm_positions, self.latest_arm_joint_state, self.max_arm_joint_change_rad
                )
                
                # Only proceed if we have limited positions (current state was available)
                if limited_arm_positions is not None:
                    arm_command = JointState()
                    arm_command.header.stamp = self.get_clock().now().to_msg()
                    arm_command.name = arm_names
                    arm_command.position = limited_arm_positions
                    arm_command.velocity = [0.0] * len(arm_names)  # Default velocity
                    arm_command.effort = [0.0] * len(arm_names)    # Default effort
                
                    self.arm_joint_pub.publish(arm_command)
                    self.get_logger().info(f'ARM COMMAND - Names: {arm_names}')
                    self.get_logger().info(f'ARM COMMAND - Positions: {limited_arm_positions}')
                    print(f"ARM COMMAND - Names: {arm_names}")
                    print(f"ARM COMMAND - Positions: {limited_arm_positions}")
                    print("---")
                else:
                    self.get_logger().warn('Skipping arm command publish - no current arm state available')
            
            # Create and publish hand joint command
            if hand_names:
                # Limit joint changes based on current joint state
                limited_hand_positions = self.limit_joint_changes(
                    hand_names, hand_positions, self.latest_hand_joint_state, self.max_hand_joint_change_rad
                )
                
                # Only proceed if we have limited positions (current state was available)
                if limited_hand_positions is not None:
                    hand_command = JointState()
                    hand_command.header.stamp = self.get_clock().now().to_msg()
                    hand_command.name = hand_names
                    hand_command.position = limited_hand_positions
                    hand_command.velocity = [0.0] * len(hand_names)  # Default velocity
                    hand_command.effort = [0.0] * len(hand_names)    # Default effort
                    
                    self.hand_joint_pub.publish(hand_command)
                    self.get_logger().info(f'HAND COMMAND - Names: {hand_names}')
                    self.get_logger().info(f'HAND COMMAND - Positions: {limited_hand_positions}')
                    print(f"HAND COMMAND - Names: {hand_names}")
                    print(f"HAND COMMAND - Positions: {limited_hand_positions}")
                    print("---")
                else:
                    self.get_logger().warn('Skipping hand command publish - no current hand state available')

            # Create and publish neck joint command
            if neck_names:
                # Limit joint changes based on current joint state
                limited_neck_positions = self.limit_joint_changes(
                    neck_names, neck_positions, self.latest_neck_joint_state, self.max_neck_joint_change_rad
                )
                
                # Only proceed if we have limited positions (current state was available)
                if limited_neck_positions is not None:
                    neck_command = JointState()
                    neck_command.header.stamp = self.get_clock().now().to_msg()
                    neck_command.name = neck_names
                    neck_command.position = limited_neck_positions
                    neck_command.velocity = [0.0] * len(neck_names)  # Default velocity
                    neck_command.effort = [0.0] * len(neck_names)    # Default effort
                    
                    self.neck_joint_pub.publish(neck_command)
                    self.get_logger().info(f'NECK COMMAND - Names: {neck_names}')
                    self.get_logger().info(f'NECK COMMAND - Positions: {limited_neck_positions}')
                    print(f"NECK COMMAND - Names: {neck_names}")
                    print(f"NECK COMMAND - Positions: {limited_neck_positions}")
                    print("---")
                else:
                    self.get_logger().warn('Skipping neck command publish - no current neck state available')
            
        except Exception as e:
            self.get_logger().error(f'Error processing JSON joint state: {e}')
            self.get_logger().error(f'Data: {data}')
    
    def arm_joint_callback(self, msg: JointState):
        """Callback for arm joint state subscription."""
        self.latest_arm_joint_state = msg
        self.arm_joint_state = msg
        self.get_logger().debug(f'Received arm joint state: {len(msg.name)} joints')
    
    def hand_joint_callback(self, msg: JointState):
        """Callback for hand joint state subscription."""
        self.latest_hand_joint_state = msg
        self.hand_joint_state = msg
        self.get_logger().debug(f'Received hand joint state: {len(msg.name)} joints')
    
    def neck_joint_callback(self, msg: JointState):
        """Callback for neck joint state subscription."""
        self.latest_neck_joint_state = msg
        self.neck_joint_state = msg
        self.get_logger().debug(f'Received neck joint state: {len(msg.name)} joints')
    
    def timer_callback(self):
        """Periodic timer callback for monitoring and maintenance."""
        # Log connection status periodically
        if self.running:
            self.get_logger().debug('ROS2 Streamer running - waiting for stdin data')
            
            # Log latest states every 5 seconds (50 timer calls at 10Hz)
            if hasattr(self, '_timer_count'):
                self._timer_count += 1
            else:
                self._timer_count = 0
            
            if self._timer_count % 50 == 0:  # Every 5 seconds
                self.log_latest_states()
                # Update joint states on gRPC server

            #self.update_grpc_joint_states()
    
    def get_latest_arm_state(self) -> Optional[JointState]:
        """Get the latest arm joint state."""
        return self.latest_arm_joint_state
    
    def get_latest_hand_state(self) -> Optional[JointState]:
        """Get the latest hand joint state."""
        return self.latest_hand_joint_state
    
    def get_latest_neck_state(self) -> Optional[JointState]:
        """Get the latest neck joint state."""
        return self.latest_neck_joint_state
    
    def limit_joint_changes(self, joint_names: List[str], target_positions: List[float], 
                          current_state: Optional[JointState], max_change: float) -> Optional[List[float]]:
        """Limit the change in joint positions to prevent sudden movements."""
        if current_state is None:
            # No current state available, return None to indicate no publishing
            self.get_logger().warn('No current joint state available, skipping publish')
            return None
        
        limited_positions = []
        changes_made = 0
        
        for i, (joint_name, target_pos) in enumerate(zip(joint_names, target_positions)):
            # Find the current position for this joint
            try:
                current_idx = current_state.name.index(joint_name)
                current_pos = current_state.position[current_idx]
                
                # Calculate the change
                change = abs(target_pos - current_pos)
                print(change)
                
                if change > max_change:
                    # Limit the change
                    if target_pos > current_pos:
                        limited_pos = current_pos + max_change
                    else:
                        limited_pos = current_pos - max_change
                    
                    limited_positions.append(limited_pos)
                    changes_made += 1
                    self.get_logger().debug(f'Limited joint {joint_name}: {current_pos:.4f} -> {target_pos:.4f} (change: {change:.4f}) -> {limited_pos:.4f}')
                else:
                    # Change is within limit
                    limited_positions.append(target_pos)
                    
            except ValueError:
                # Joint not found in current state, use target position
                limited_positions.append(target_pos)
                self.get_logger().debug(f'Joint {joint_name} not found in current state, using target: {target_pos:.4f}')
        
        if changes_made > 0:
            self.get_logger().info(f'Limited changes for {changes_made} joints (max change: {max_change:.4f} rad)')
        
        return limited_positions
    
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
        if self.latest_arm_joint_state:
            self.get_logger().info(f'Latest ARM STATE - {len(self.latest_arm_joint_state.name)} joints: {self.latest_arm_joint_state.name[:3]}...')
        else:
            self.get_logger().info('Latest ARM STATE - No data received yet')
            
        if self.latest_hand_joint_state:
            self.get_logger().info(f'Latest HAND STATE - {len(self.latest_hand_joint_state.name)} joints: {self.latest_hand_joint_state.name[:3]}...')
        else:
            self.get_logger().info('Latest HAND STATE - No data received yet')

        if self.latest_neck_joint_state:
            self.get_logger().info(f'Latest NECK STATE - {len(self.latest_neck_joint_state.name)} joints: {self.latest_neck_joint_state.name[:3]}...')
        else:
            self.get_logger().info('Latest NECK STATE - No data received yet')
    
    def shutdown(self):
        """Clean shutdown of the node."""
        self.running = False
        self.get_logger().info('ROS2 Streamer shutdown complete')


def main(args=None):
    rclpy.init(args=args)
    
    streamer = ROS2Streamer()
    
    try:
        rclpy.spin(streamer)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.shutdown()
        streamer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()