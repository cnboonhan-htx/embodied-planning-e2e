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

# RH was here to add the below
from collections import deque
action_queue = deque()
prev_action_values = None
previous_joint_updates = None
# RH was here to add the above
logger = logging.getLogger(__name__)

# Using grpcurl instead of gRPC imports

FPS = 10
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

# RH was here to add the below
def map_action_values_to_joints(action_values, dataset):
    """Map action values from predict_action to joint names for gRPC."""
    action_names = dataset.features["action"]["names"]  # uses the dataset ordering
    joint_updates = {}

    # convert to 1D tensor/array
    if hasattr(action_values, "detach"):
        action_values = action_values.detach().cpu()
    if hasattr(action_values, "numpy"):
        action_arr = action_values.numpy()
    else:
        action_arr = action_values

    if action_arr.ndim > 1:
        action_arr = action_arr[0]

    for i, full_name in enumerate(action_names):
        if i < len(action_arr):
            base_name = full_name.replace(".pos", "")
            joint_updates[base_name] = float(action_arr[i])

    return joint_updates
# RH was here to add the above

# RH was here to comment the below
# def map_action_values_to_joints(action_values):
#     """Map action values from predict_action to joint names for gRPC."""
#     joint_names = list(ROBOT_JOINT_MAPPING.keys())
#     joint_updates = {}
#     print(f"Joint names: {joint_names}")
#     print(f"Action values: {action_values}")
    
#     # Handle different tensor shapes
#     if hasattr(action_values, 'shape'):
#         if len(action_values.shape) == 0:  # 0-d tensor (scalar)
#             print("Warning: action_values is a scalar, cannot map to joints")
#             return {}
#         elif len(action_values.shape) == 1:  # 1-d tensor
#             action_tensor = action_values
#         else:  # Multi-dimensional tensor, take first element
#             action_tensor = action_values[0]
#     else:
#         action_tensor = action_values
    
#     # Map values to joint names
#     for i, joint_name in enumerate(joint_names):
#         if i < len(action_tensor):
#             joint_updates[joint_name] = float(action_tensor[i])

#     print(f"Joint updates: {joint_updates}")
#     return joint_updates
# RH was here to comment the above

# Load policy configuration
policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=POLICY_REPO_NAME)
dataset_root = Path(f"/home/cnboonhan/data_collection/{DATA_REPO_NAME}").expanduser()
dataset_exists = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()
dataset = LeRobotDataset(DATA_REPO_NAME, root=str(dataset_root), force_cache_sync=True)
policy: PreTrainedPolicy = make_policy(policy_cfg, ds_meta=dataset.meta)
policy.cuda()
policy.eval()
policy.reset()

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
robot = ViserFollower(camera_config)
robot.connect()
device = torch.device(policy.config.device)

print(f"🔌 Will use grpcurl to connect to: {ROBOT_SERVER_GRPC_URL}")
print("✅ Ready to send joint updates via grpcurl!")

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg,
    pretrained_path=POLICY_REPO_NAME,
    dataset_stats=dataset.meta.stats, # RH was here to add this
)

policy.reset()
preprocessor.reset()
postprocessor.reset()
    
start_time = time.perf_counter()
loop_counter = 0
target_frame_time = 1.0 / FPS  # Target time per frame in seconds

input("Initializing robot to starting position... Press Enter to continue...")

# episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == 2)
# actions = episode_frames.select_columns("action")
# action_array = actions[0]["action"]
# action = {}
# for i, name in enumerate(dataset.features["action"]["names"]):
#     action[name] = action_array[i]

# # Map action to joint updates
# joint_updates = map_action_values_to_joints(action_array, dataset) # RH was here to add dataset variable
# send_joint_updates_grpc(joint_updates)
# previous_joint_updates = joint_updates
# input("Robot initialized. Press Enter to start inference loop...")
    
# RH was here to add the below
def split_action_values(action_values):
    if hasattr(action_values, "detach"):
        a = action_values.detach().cpu()
    else:
        a = torch.tensor(action_values)

    # Collapse leading batch dims if any
    while a.dim() > 2:
        a = a.squeeze(0)

    if a.dim() == 2:
        # shape (n_steps, n_joints)
        steps = [a[i].clone() for i in range(a.shape[0])]
    elif a.dim() == 1:
        # flattened: try to infer n_joints from dataset features
        n_joints = len(dataset.features["action"]["names"])
        if a.numel() % n_joints != 0:

            # fallback: treat it as single step
            steps = [a.clone()]
        else:
            n_steps = a.numel() // n_joints
            steps = [a[i * n_joints : (i + 1) * n_joints].clone() for i in range(n_steps)]
    else:
        # unexpected shape: try first row
        steps = [a[0].clone()]

    return steps
# RH was here to add the above

# counter = 1 # RH was here to comment this out
try:
    while True:
        loop_start = time.perf_counter()
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        # observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")

        # RH was here to add the below
        missing_cams = []
        for cam_key in ["headcam", "wristcam"]:
            if cam_key not in obs_processed or obs_processed[cam_key] is None:
                missing_cams.append(cam_key)

        if missing_cams:
            print(f"⚠️ Missing cameras {missing_cams}")
            continue
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")
        # RH was here to add the above

        # action_values = predict_action( # RH was here to comment this out
        if not action_queue: # RH was here to add this line
            # input("Next action? Press Enter to continue...")
            action_values = predict_action( # RH was here to add this line
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=TASK,
                robot_type="a2",
            )

# RH was here to add the below
        if hasattr(action_values, "detach"):
            action_values = action_values.detach().cpu().to(torch.float32)

        # Clamp to safe teleop range, consistent with ViserLeader normalization
        TELEOP_CLAMP = 100.0  # since ViserLeader scales all joints to [-100, 100]
        action_values = action_values.clamp(-TELEOP_CLAMP, TELEOP_CLAMP)
        
        # Optional debug print
        print(f"Predicted normalized action_values: min={action_values.min():.2f}, max={action_values.max():.2f}, mean={action_values.mean():.2f}")

        # Map and send as usual
        joint_updates = map_action_values_to_joints(action_values, dataset)
# RH was here to add the above

        # joint_updates = map_action_values_to_joints(action_values) # RH was here to comment this out

        # Do an exponential smoothing of joint updates with previous_joint_updates
        if previous_joint_updates:
            alpha = 0.5  # RH was here to change smoothing factor from 0.2 to 0.5
            for joint in joint_updates:
                if joint in previous_joint_updates:
                    joint_updates[joint] = alpha * joint_updates[joint] + (1 - alpha) * previous_joint_updates[joint]
        send_joint_updates_grpc(joint_updates)

        previous_joint_updates = joint_updates
        loop_duration = time.perf_counter() - loop_start
        sleep_time = target_frame_time - loop_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
        # counter += 100 the above
        # input("Press Enter for next step...")

        # while policy._action_queue:
        #     input("Press Enter to send next action...")
        #     loop_start = time.perf_counter()
        #     print(f"⏭️ Processing queued action, {len(policy._action_queue)} left")
        #     action = policy._action_queue.popleft().squeeze(0).to("cpu")
        #     joint_updates = map_action_values_to_joints(action)
        #     send_joint_updates_grpc(joint_updates)
        #     loop_duration = time.perf_counter() - loop_start
        #     sleep_time = target_frame_time - loop_duration
        #     if sleep_time > 0:
        #         time.sleep(sleep_time)

        
        
    
except KeyboardInterrupt:
    print("\nInference stopped by user")

finally:
    print("Disconnecting robot...")
    robot.disconnect()
