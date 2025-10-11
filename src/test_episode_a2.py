#!/usr/bin/env python
import time
import logging
import torch
import subprocess
import json
import argparse
from typing import Optional
from pathlib import Path
from utils_a2 import ViserFollower
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from utils_a2 import ROBOT_JOINT_MAPPING, camera_config, DATA_REPO_NAME

logger = logging.getLogger(__name__)

# Using grpcurl instead of gRPC imports

FPS = 15
ROBOT_SERVER_GRPC_URL = "localhost:5000"

def parse_args():
    parser = argparse.ArgumentParser(description="Replay episode from dataset")
    parser.add_argument("--episode", type=int, required=True, help="Episode number to replay")
    return parser.parse_args()

args = parse_args()
EPISODE_NUMBER = args.episode
print(f"Replaying episode {EPISODE_NUMBER}")

def send_joint_updates_grpc(joint_updates):
    """Send joint updates to the robot server using grpcurl."""
    try:
        if joint_updates:
            # Prepare the grpcurl command
            command = [
                "grpcurl",
                "-plaintext",
                "-d", json.dumps({"joint_updates": joint_updates}),
                ROBOT_SERVER_GRPC_URL,
                "rosbot_api.RobotApiService/UpdateJoints"
            ]
            
            # Execute the command
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print(f"🔄 Sent {len(joint_updates)} joint updates via grpcurl")
                if result.stdout:
                    response = json.loads(result.stdout)
                    print(f"Response: {response}")
                return True
            else:
                print(f"❌ grpcurl error: {result.stderr}")
                return False
        else:
            print("⏭️ No joint updates to send")
            return True
            
    except subprocess.TimeoutExpired:
        print("❌ grpcurl timeout")
        return False
    except Exception as e:
        print(f"❌ Error sending joints: {e}")
        return False

def map_action_values_to_joints(action_values):
    """Map action values from predict_action to joint names for gRPC."""
    joint_names = list(ROBOT_JOINT_MAPPING.keys())
    
    joint_updates = {}
    print(f"Action values: {action_values}")
    
    # Handle different tensor shapes
    if hasattr(action_values, 'shape'):
        if len(action_values.shape) == 0:  # 0-d tensor (scalar)
            print("Warning: action_values is a scalar, cannot map to joints")
            return {}
        elif len(action_values.shape) == 1:  # 1-d tensor
            action_tensor = action_values
        else:  # Multi-dimensional tensor, take first element
            action_tensor = action_values[0]
    else:
        action_tensor = action_values
    
    # Map values to joint names
    for i, joint_name in enumerate(joint_names):
        if i < len(action_tensor):
            joint_updates[joint_name] = float(action_tensor[i])
    
    return joint_updates

# Load dataset for episode replay
dataset_root = Path(f"/home/cnboonhan/data_collection/{DATA_REPO_NAME}").expanduser()
dataset_exists = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()
dataset = LeRobotDataset(DATA_REPO_NAME, root=str(dataset_root), episodes=[EPISODE_NUMBER])

# Filter dataset to only include frames from the specified episode
episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_NUMBER)
actions = episode_frames.select_columns("action")

print(f"📊 Loaded episode {EPISODE_NUMBER} with {len(episode_frames)} frames")
print(f"🔌 Will use grpcurl to connect to: {ROBOT_SERVER_GRPC_URL}")
print("✅ Ready to replay episode!")

robot = ViserFollower(camera_config)
robot.connect()
    
start_time = time.perf_counter()
target_frame_time = 1.0 / FPS  # Target time per frame in seconds
    
try:
    print(f"🎬 Starting replay of episode {EPISODE_NUMBER} with {len(episode_frames)} frames")
    
    for idx in range(len(episode_frames)):
        loop_start = time.perf_counter()
        
        # Get action from the episode data
        action_array = actions[idx]["action"]
        action = {}
        for i, name in enumerate(dataset.features["action"]["names"]):
            action[name] = action_array[i]
        
        print(f"Frame {idx+1}/{len(episode_frames)}: {action}")
        
        # Map action to joint updates
        joint_updates = map_action_values_to_joints(action_array)
        send_joint_updates_grpc(joint_updates)
        
        loop_duration = time.perf_counter() - loop_start
        sleep_time = target_frame_time - loop_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    print(f"✅ Episode {EPISODE_NUMBER} replay completed!")
    
except KeyboardInterrupt:
    print("\nReplay stopped by user")

finally:
    print("Disconnecting robot...")
    robot.disconnect()
