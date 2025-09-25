import json
import subprocess
import sys
import time
from typing import Dict, Any, Optional
from pathlib import Path
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

def get_robot_info(api_port: Optional[int] = None) -> Dict[str, Any]:
    """Query robot info to get joint limits"""
    command = [
        "grpcurl",
        "-plaintext",
        "-format", "json",
        f"localhost:{api_port}",
        "rosbot_api.RobotApiService/GetRobotInfo"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Error getting robot info: {e}", file=sys.stderr)
        return {}

def normalize_joint_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize joint value from [min_val, max_val] to [-100, 100]"""
    if max_val == min_val:
        return 0.0
    
    # Normalize to [0, 1] first
    normalized = (value - min_val) / (max_val - min_val)
    
    # Scale to [-100, 100]
    return (normalized * 200) - 100

def stream_joint_data(robot: SO101Follower, update_frequency: float = 100.0, api_port: Optional[int] = None) -> None:
    # Get robot info to obtain joint limits
    robot_info = get_robot_info(api_port)
    joint_limits = robot_info.get('joint_limits', {})
    
    print(f"Robot info retrieved. Joint limits: {joint_limits}", file=sys.stderr)
    
    command = [
        "grpcurl",
        "-plaintext",
        "-format", "json",
        "-d", json.dumps({"update_frequency": update_frequency}),
        f"localhost:{api_port}",
        "rosbot_api.RobotApiService/StreamJointData"
    ]
    
    try:
        print(f"Starting joint data stream with frequency: {update_frequency} Hz", file=sys.stderr)
        print(f"Connecting to API on port: {api_port}", file=sys.stderr)
        print("Press Ctrl+C to stop streaming", file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        
        # Execute the streaming command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        buffer = ""
        brace_count = 0
        
        for line in process.stdout:
            line = line.strip()
            if line:
                buffer += line
                # Count braces to detect complete JSON objects
                brace_count += line.count('{') - line.count('}')
                
                # If we have a complete JSON object (balanced braces)
                if brace_count == 0 and buffer:
                    try:
                        # Parse the complete JSON object
                        json_data = json.loads(buffer)
                        
                        # Extract joint positions
                        joint_positions = json_data['joint_positions']
                        
                        # Normalize each joint value to [-100, 100] range
                        normalized_joints = {}
                        for joint_id, value in joint_positions.items():
                            if joint_id in joint_limits:
                                limits = joint_limits[joint_id]
                                min_val = limits.get('lower', 0)
                                max_val = limits.get('upper', 1)
                                normalized_value = normalize_joint_value(value, min_val, max_val)
                                normalized_joints[joint_id] = normalized_value
                            else:
                                # If no limits available, use raw value
                                normalized_joints[joint_id] = value
                        
                        # Print normalized joint values
                        print("Normalized Joint Values (-100 to 100):")
                        for joint_id, normalized_value in normalized_joints.items():
                            print(f"  Joint {joint_id}: {normalized_value:.2f}")
                        
                        # Also print original values for reference
                        print("Original Joint Values:")
                        for joint_id, value in joint_positions.items():
                            print(f"  Joint {joint_id}: {value:.2f}")
                        print("-" * 30)
                        
                        # Send normalized values to robot
                        robot.send_action({
                            'shoulder_pan.pos': -normalized_joints.get('1', 0),
                            'shoulder_lift.pos': normalized_joints.get('2', 0),
                            'elbow_flex.pos': normalized_joints.get('3', 0),
                            'wrist_flex.pos': normalized_joints.get('4', 0),
                            'wrist_roll.pos': normalized_joints.get('5', 0),
                            'gripper.pos': normalized_joints.get('6', 0)})
                        
                        # Reset buffer for next JSON object
                        buffer = ""
                        
                    except json.JSONDecodeError as e:
                        # If it's not valid JSON, print as-is
                        print(f"Non-JSON data received: {buffer}")
                        print(f"JSON decode error: {e}")
                        buffer = ""
                        brace_count = 0
                    
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user", file=sys.stderr)
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"Error in streaming: {e}", file=sys.stderr)
        if 'process' in locals():
            process.terminate()


def main(update_frequency: float = 100.0, api_port: Optional[int] = None):
    robot_config = SO101FollowerConfig(
        port="/dev/so100",
        id="so100",
    )
    robot = SO101Follower(robot_config)
    robot.connect()
    stream_joint_data(robot, update_frequency, api_port)


if __name__ == "__main__":
    main(update_frequency=100.0, api_port=5000)
