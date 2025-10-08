#!/usr/bin/env python
import time
import logging
import torch
import subprocess
import json
from typing import Optional
from pathlib import Path
from utils_a2 import ViserFollower
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.processor.factory import make_default_processors
from lerobot.utils.control_utils import (
    predict_action,
)
from lerobot.utils.utils import (
    get_safe_torch_device
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from utils_a2 import ROBOT_JOINT_MAPPING, camera_config, TASK_DESCRIPTION, POLICY_REPO_NAME, DATA_REPO_NAME

logger = logging.getLogger(__name__)

# Using grpcurl instead of gRPC imports

FPS = 0.5
TASK = TASK_DESCRIPTION
ROBOT_SERVER_GRPC_URL = "localhost:5000"
print(f"Loading policy from {POLICY_REPO_NAME}")

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

# Load policy configuration
policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=POLICY_REPO_NAME)
dataset_root = Path(f"/home/cnboonhan/data_collection/{DATA_REPO_NAME}").expanduser()
dataset_exists = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()
dataset = LeRobotDataset(DATA_REPO_NAME, root=str(dataset_root))
policy: PreTrainedPolicy = make_policy(policy_cfg, ds_meta=dataset.meta)

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
robot = ViserFollower(camera_config)
robot.connect()
device = torch.device(policy.config.device)

print(f"🔌 Will use grpcurl to connect to: {ROBOT_SERVER_GRPC_URL}")
print("✅ Ready to send joint updates via grpcurl!")

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg,
    pretrained_path=POLICY_REPO_NAME,
)

policy.reset()
preprocessor.reset()
postprocessor.reset()
    
start_time = time.perf_counter()
loop_counter = 0
target_frame_time = 1.0 / FPS  # Target time per frame in seconds
    
try:
    loop_counter = 0

    while True:
        loop_start = time.perf_counter()
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=TASK,
            robot_type="a2",
        )
        if loop_counter == 8:
            joint_updates = map_action_values_to_joints(action_values)
            send_joint_updates_grpc(joint_updates)
            loop_duration = time.perf_counter() - loop_start
            sleep_time = target_frame_time - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            loop_counter = 0
        else:
            loop_counter += 1
        
    
except KeyboardInterrupt:
    print("\nInference stopped by user")

finally:
    print("Disconnecting robot...")
    robot.disconnect()
