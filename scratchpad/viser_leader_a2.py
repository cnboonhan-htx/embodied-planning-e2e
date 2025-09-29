#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import subprocess
import time
from typing import Dict, Any, Optional

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.teleoperators.teleoperator import Teleoperator

logger = logging.getLogger(__name__)


class ViserLeader():
    name = "viser_leader"

    def __init__(self, api_port: int = 5000):
        self.api_port = api_port
        self.joint_limits = {}
        self._get_robot_info()

    @property
    def action_features(self) -> dict[str, type]:
        return {
            # f"idx13_left_arm_joint1.pos": float, 
            # f"idx14_left_arm_joint2.pos": float, 
            # f"idx15_left_arm_joint3.pos": float, 
            # f"idx16_left_arm_joint4.pos": float, 
            # f"idx17_left_arm_joint5.pos": float, 
            # f"idx18_left_arm_joint6.pos": float, 
            # f"idx19_left_arm_joint7.pos": float, 
            f"idx20_right_arm_joint1.pos": float, 
            f"idx21_right_arm_joint2.pos": float, 
            f"idx22_right_arm_joint3.pos": float, 
            f"idx23_right_arm_joint4.pos": float, 
            f"idx24_right_arm_joint5.pos": float, 
            f"idx25_right_arm_joint6.pos": float, 
            f"idx26_right_arm_joint7.pos": float,
            # f"left_thumb_0.pos": float,
            # f"left_thumb_1.pos": float,
            # f"left_index.pos": float,
            # f"left_middle.pos": float,
            # f"left_ring.pos": float,
            # f"left_pinky.pos": float,
            # f"right_thumb_0.pos": float,
            # f"right_thumb_1.pos": float,
            # f"right_index.pos": float,
            # f"right_middle.pos": float,
            # f"right_ring.pos": float,
            # f"right_pinky.pos": float
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def _get_robot_info(self) -> None:
        """Query robot info to get joint limits"""
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
            self.joint_limits = robot_info.get('joint_limits', {})
            logger.info(f"Retrieved joint limits: {self.joint_limits}")
        except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Error getting robot info: {e}")
            self.joint_limits = {}

    def _normalize_joint_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize joint value from [min_val, max_val] to [-100, 100]"""
        if max_val == min_val:
            return 0.0
        
        # Normalize to [0, 1] first
        normalized = (value - min_val) / (max_val - min_val)
        
        # Scale to [-100, 100]
        return (normalized * 200) - 100

    def get_action(self) -> dict[str, float]:
        """Get current joint positions and normalize them"""
        # Get raw joint positions
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
            action = {}
            
            # Joint mapping from gRPC response keys to expected joint names
            joint_mapping = {
                # "idx13_left_arm_joint1": "idx13_left_arm_joint1",
                # "idx14_left_arm_joint2": "idx14_left_arm_joint2", 
                # "idx15_left_arm_joint3": "idx15_left_arm_joint3",
                # "idx16_left_arm_joint4": "idx16_left_arm_joint4",
                # "idx17_left_arm_joint5": "idx17_left_arm_joint5",
                # "idx18_left_arm_joint6": "idx18_left_arm_joint6",
                # "idx19_left_arm_joint7": "idx19_left_arm_joint7",
                "idx20_right_arm_joint1": "idx20_right_arm_joint1",
                "idx21_right_arm_joint2": "idx21_right_arm_joint2",
                "idx22_right_arm_joint3": "idx22_right_arm_joint3",
                "idx23_right_arm_joint4": "idx23_right_arm_joint4",
                "idx24_right_arm_joint5": "idx24_right_arm_joint5",
                "idx25_right_arm_joint6": "idx25_right_arm_joint6",
                "idx26_right_arm_joint7": "idx26_right_arm_joint7",
                # "left_thumb_0": "left_thumb_0",
                # "left_thumb_1": "left_thumb_1",
                # "left_index": "left_index",
                # "left_middle": "left_middle",
                # "left_ring": "left_ring",
                # "left_pinky": "left_pinky",
                # "right_thumb_0": "right_thumb_0",
                # "right_thumb_1": "right_thumb_1",
                # "right_index": "right_index",
                # "right_middle": "right_middle",
                # "right_ring": "right_ring",
                # "right_pinky": "right_pinky"
            }
            
            for joint_name, position in joint_positions.items():
                # Map the joint name to the expected format
                if joint_name in joint_mapping:
                    mapped_joint_name = joint_mapping.get(joint_name, joint_name)
                    action[f"{mapped_joint_name}.pos"] = position
            for joint_name, position in custom_joint_positions.items():
                # Map the joint name to the expected format
                if joint_name in joint_mapping:
                    mapped_joint_name = joint_mapping.get(joint_name, joint_name)
                    action[f"{mapped_joint_name}.pos"] = position
            
            return action 
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Error getting robot observation: {e}")
            return {}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # TODO
        logger.info(f"{self} disconnected.")