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
        return {f"shoulder_pan.pos": float, f"shoulder_lift.pos": float, f"elbow_flex.pos": float, f"wrist_flex.pos": float, f"wrist_roll.pos": float, f"gripper.pos": float}

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

    def _get_single_joint_data(self) -> Dict[str, float]:
        """Get a single joint data reading using grpcurl"""
        command = [
            "grpcurl",
            "-plaintext",
            "-format", "json",
            f"localhost:{self.api_port}",
            "rosbot_api.RobotApiService/GetRobotInfo"
        ]
        
        try:
            # Run grpcurl with timeout to get a single reading
            result = subprocess.run(command, capture_output=True, text=True, timeout=2)
            
            # Parse the output the same way as so100_controller.py
            buffer = ""
            brace_count = 0
            
            for line in result.stdout:
                line = line.strip()
                if line:
                    buffer += line
                    # Count braces to detect complete JSON objects
                    brace_count += line.count('{') - line.count('}')
                    
                    # If we have a complete JSON object (balanced braces)
                    if brace_count == 0 and buffer:
                        try:
                            json_data = json.loads(buffer)
                            return json_data.get('joint_positions', {})
                        except json.JSONDecodeError:
                            buffer = ""
                            brace_count = 0
                            continue
            
            logger.warning("No valid joint data received")
            return {}
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Error getting joint data: {e}")
            return {}

    def get_action(self) -> dict[str, float]:
        """Get current joint positions and normalize them"""
        # Get raw joint positions
        joint_positions = self._get_single_joint_data()
        
        if not joint_positions:
            logger.warning("No joint data available")
            return {}
        
        # Normalize joint values and map to joint names
        normalized_actions = {}
        joint_mapping = {
            '1': 'shoulder_pan.pos',
            '2': 'shoulder_lift.pos', 
            '3': 'elbow_flex.pos',
            '4': 'wrist_flex.pos',
            '5': 'wrist_roll.pos',
            '6': 'gripper.pos'
        }
        
        for joint_id, value in joint_positions.items():
            if joint_id in self.joint_limits:
                limits = self.joint_limits[joint_id]
                min_val = limits.get('lower', 0)
                max_val = limits.get('upper', 1)
                normalized_value = self._normalize_joint_value(value, min_val, max_val)
                
                # Apply same sign inversion as so100_controller.py
                if joint_id == '1':  # shoulder_pan
                    normalized_value = -normalized_value
                    
                joint_name = joint_mapping.get(joint_id, f"{joint_id}.pos")
                normalized_actions[joint_name] = normalized_value
            else:
                # If no limits available, use raw value
                joint_name = joint_mapping.get(joint_id, f"{joint_id}.pos")
                normalized_actions[joint_name] = value
        
        return normalized_actions

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # TODO
        logger.info(f"{self} disconnected.")