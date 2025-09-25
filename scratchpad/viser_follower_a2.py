import json
import logging
import subprocess
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position

logger = logging.getLogger(__name__)


class ViserFollower(Robot):

    name = "viser_follower"

    def __init__(self):
        self.cameras = {}
        self.id = "viser_follower"
        self.api_port = 5001
        # super().__init__()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"idx13_left_arm_joint1.pos": float, 
            f"idx14_left_arm_joint2.pos": float, 
            f"idx15_left_arm_joint3.pos": float, 
            f"idx16_left_arm_joint4.pos": float, 
            f"idx17_left_arm_joint5.pos": float, 
            f"idx18_left_arm_joint6.pos": float, 
            f"idx19_left_arm_joint7.pos": float, 
            f"idx20_right_arm_joint1.pos": float, 
            f"idx21_right_arm_joint2.pos": float, 
            f"idx22_right_arm_joint3.pos": float, 
            f"idx23_right_arm_joint4.pos": float, 
            f"idx24_right_arm_joint5.pos": float, 
            f"idx25_right_arm_joint6.pos": float, 
            f"idx26_right_arm_joint7.pos": float,
            f"left_thumb_0.pos": float,
            f"left_thumb_1.pos": float,
            f"left_index.pos": float,
            f"left_middle.pos": float,
            f"left_ring.pos": float,
            f"left_pinky.pos": float,
            f"right_thumb_0.pos": float,
            f"right_thumb_1.pos": float,
            f"right_index.pos": float,
            f"right_middle.pos": float,
            f"right_ring.pos": float,
            f"right_pinky.pos": float
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"{self} connected.")

    @property
    def calibrate(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        """Get current robot joint positions using gRPC"""
        command = [
            "grpcurl",
            "-plaintext",
            "-format", "json",
            f"localhost:{self.api_port}",
            "rosbot_api.RobotApiService/GetRobotInfo"
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
            robot_info = json.loads(result.stdout)
            joint_positions = robot_info.get('joint_positions', {})
            custom_joint_positions = robot_info.get('custom_joint_positions', {})
            
            # Convert joint positions to observation format with observation.(joint_name).pos convention
            observation = {}
            
            # Joint mapping from gRPC response keys to expected joint names
            joint_mapping = {
                "idx13_left_arm_joint1": "idx13_left_arm_joint1",
                "idx14_left_arm_joint2": "idx14_left_arm_joint2", 
                "idx15_left_arm_joint3": "idx15_left_arm_joint3",
                "idx16_left_arm_joint4": "idx16_left_arm_joint4",
                "idx17_left_arm_joint5": "idx17_left_arm_joint5",
                "idx18_left_arm_joint6": "idx18_left_arm_joint6",
                "idx19_left_arm_joint7": "idx19_left_arm_joint7",
                "idx20_right_arm_joint1": "idx20_right_arm_joint1",
                "idx21_right_arm_joint2": "idx21_right_arm_joint2",
                "idx22_right_arm_joint3": "idx22_right_arm_joint3",
                "idx23_right_arm_joint4": "idx23_right_arm_joint4",
                "idx24_right_arm_joint5": "idx24_right_arm_joint5",
                "idx25_right_arm_joint6": "idx25_right_arm_joint6",
                "idx26_right_arm_joint7": "idx26_right_arm_joint7",
                "left_thumb_0": "left_thumb_0",
                "left_thumb_1": "left_thumb_1",
                "left_index": "left_index",
                "left_middle": "left_middle",
                "left_ring": "left_ring",
                "left_pinky": "left_pinky",
                "right_thumb_0": "right_thumb_0",
                "right_thumb_1": "right_thumb_1",
                "right_index": "right_index",
                "right_middle": "right_middle",
                "right_ring": "right_ring",
                "right_pinky": "right_pinky"
            }
            
            for joint_name, position in joint_positions.items():
                # Map the joint name to the expected format
                if joint_name in joint_mapping:
                    mapped_joint_name = joint_mapping.get(joint_name, joint_name)
                    observation[f"{mapped_joint_name}.pos"] = position
            for joint_name, position in custom_joint_positions.items():
                # Map the joint name to the expected format
                if joint_name in joint_mapping:
                    mapped_joint_name = joint_mapping.get(joint_name, joint_name)
                    observation[f"{mapped_joint_name}.pos"] = position
            
            print(f"Retrieved joint positions: {joint_positions}")
            return observation
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Error getting robot observation: {e}")
            return {}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        pass

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        logger.info(f"{self} disconnected.")
