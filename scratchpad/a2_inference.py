#!/usr/bin/env python
import time
import torch
from typing import Optional
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.processor.factory import make_default_processors
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from viser_follower_a2 import ViserFollower
from lerobot.utils.control_utils import (
    predict_action,
    get_safe_torch_device,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts

POLICY_PATH = "cnboonhan-htx/a2_act_wave"
REPO_NAME = "cnboonhan-htx/a2-wave-2909"
FPS = 15
TASK = "wave"
print(f"Loading policy from {POLICY_PATH}")

# Load policy configuration
camera_config = {"front": RealSenseCameraConfig("211622068536", ColorMode.RGB, False, Cv2Rotation.NO_ROTATION, 0, width=640, height=480, fps=FPS)  }
policy_cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=POLICY_PATH)
dataset_root = Path(f"~/data_collection/{REPO_NAME}").expanduser()
dataset_exists = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()
dataset = LeRobotDataset(REPO_NAME, root=str(dataset_root))
policy: PreTrainedPolicy = make_policy(policy_cfg, ds_meta=dataset.meta)

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
camera = RealSenseCamera(camera_config["front"])
robot = ViserFollower(camera_config)
robot.connect()
device = torch.device(policy.config.device)

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg,
    pretrained_path=POLICY_PATH,
)

policy.reset()
preprocessor.reset()
postprocessor.reset()
    
start_time = time.perf_counter()
    
try:
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
            robot_type=robot.robot_type,
        )
    
except KeyboardInterrupt:
    print("\nInference stopped by user")

finally:
    print("Disconnecting robot...")
    robot.disconnect()


if __name__ == "__main__":
    # Example usage - you'll need to provide your robot config
    import argparse
    
    parser = argparse.ArgumentParser(description="Run policy inference")
    parser.add_argument("--policy_path", required=True, help="Path to pretrained policy")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=float, help="Duration in seconds (infinite if not specified)")
    parser.add_argument("--task", help="Task description")
    parser.add_argument("--device", help="Device to run on (cuda/cpu)")
    
    args = parser.parse_args()
    