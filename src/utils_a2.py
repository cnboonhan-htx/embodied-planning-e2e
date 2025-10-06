from functools import cached_property
import json
import logging
import subprocess
import time
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import (
    RealSenseCameraConfig,
)  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.cameras.configs import ColorMode, Cv2Rotation

logger = logging.getLogger(__name__)

TASK_DESCRIPTION = "My task description"
POLICY_REPO_NAME = "cnboonhan-htx/a2-pnp-0610-right-hand"
DATA_REPO_NAME = "cnboonhan-htx/a2-pnp-0610-right-hand"
ROBOT_JOINT_MAPPING = {
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
    "right_thumb_0": "right_thumb_0",
    "right_thumb_1": "right_thumb_1",
    "right_index": "right_index",
    "right_middle": "right_middle",
    "right_ring": "right_ring",
    "right_pinky": "right_pinky",
}

# Camera configuration
camera_config = {
    "front": RealSenseCameraConfig(
        "211622068536",
        ColorMode.RGB,
        False,
        Cv2Rotation.NO_ROTATION,
        0,
        width=640,
        height=480,
        fps=30,
    ),
    "headcam": OpenCVCameraConfig(
        index_or_path=8,
        width=640,
        height=480,
        fps=30,
        color_mode=ColorMode.RGB,
    ),
    "phone": OpenCVCameraConfig(
        index_or_path=9,
        width=640,
        height=480,
        fps=30,
        color_mode=ColorMode.RGB,
    )
}


class ViserFollowerConfig:
    cameras: dict[str, CameraConfig]

    def __init__(self, cameras: dict[str, CameraConfig]):
        self.cameras = cameras


class ViserLeader:
    name = "viser_leader"

    def __init__(self, api_port: int = 5000):
        self.api_port = api_port
        self.joint_limits = {}
        self._get_robot_info()

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in ROBOT_JOINT_MAPPING.keys()}

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
            "-format",
            "json",
            f"localhost:{self.api_port}",
            "rosbot_api.RobotApiService/GetRobotInfo",
        ]

        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, timeout=5
            )
            robot_info = json.loads(result.stdout)
            self.joint_limits = robot_info.get("joint_limits", {})
            logger.info(f"Retrieved joint limits: {self.joint_limits}")
        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            subprocess.TimeoutExpired,
        ) as e:
            logger.warning(f"Error getting robot info: {e}")
            self.joint_limits = {}

    def _normalize_joint_value(
        self, value: float, min_val: float, max_val: float
    ) -> float:
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
            "-format",
            "json",
            f"localhost:{self.api_port}",
            "rosbot_api.RobotApiService/GetRobotInfo",
        ]

        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, timeout=5
            )
            robot_info = json.loads(result.stdout)
            joint_positions = robot_info.get("joint_positions", {})
            custom_joint_positions = robot_info.get("custom_joint_positions", {})

            # Convert joint positions to observation format with observation.(joint_name).pos convention
            action = {}

            # Joint mapping from gRPC response keys to expected joint names
            joint_mapping = ROBOT_JOINT_MAPPING

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

        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            subprocess.TimeoutExpired,
        ) as e:
            logger.warning(f"Error getting robot observation: {e}")
            return {}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        logger.info(f"{self} disconnected.")


class ViserFollower(Robot):
    name = "viser_follower"

    def __init__(self, cameras: dict[str, CameraConfig]):
        self.config = ViserFollowerConfig(cameras=cameras)
        self.cameras = make_cameras_from_configs(cameras)
        self.id = "viser_follower"
        self.api_port = 5001

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in ROBOT_JOINT_MAPPING.keys()}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        # Connect cameras
        for cam_name, cam in self.cameras.items():
            try:
                cam.connect()
                logger.info(f"Connected camera {cam_name}")
            except Exception as e:
                logger.warning(f"Failed to connect camera {cam_name}: {e}")

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
            "-format",
            "json",
            f"localhost:{self.api_port}",
            "rosbot_api.RobotApiService/GetRobotInfo",
        ]

        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, timeout=5
            )
            robot_info = json.loads(result.stdout)
            joint_positions = robot_info.get("joint_positions", {})
            custom_joint_positions = robot_info.get("custom_joint_positions", {})

            # Convert joint positions to observation format with observation.(joint_name).pos convention
            observation = {}

            # Joint mapping from gRPC response keys to expected joint names
            joint_mapping = ROBOT_JOINT_MAPPING

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

            # Add camera data to observation
            for cam_key, cam in self.cameras.items():
                start = time.perf_counter()
                max_retries = 3
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        observation[cam_key] = cam.async_read()
                        dt_ms = (time.perf_counter() - start) * 1e3
                        logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
                        break  # Success, exit retry loop
                    except TimeoutError as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(
                                f"Timeout reading camera {cam_key}, retrying ({retry_count}/{max_retries}): {e}"
                            )
                            time.sleep(0.01)  # Brief delay before retry
                        else:
                            logger.error(
                                f"Failed to read camera {cam_key} after {max_retries} attempts: {e}"
                            )
                            # Skip this camera - don't add to observation
                    except Exception as e:
                        logger.warning(f"Error reading camera {cam_key}: {e}")
                        break  # Don't retry for other errors

            return observation

        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            subprocess.TimeoutExpired,
        ) as e:
            logger.warning(f"Error getting robot observation: {e}")
            return {}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        pass

    def disconnect(self):
        logger.info(f"{self} disconnected.")


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError(
                "Choose a policy, a teleoperator or both to control the robot"
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(
            f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps})."
        )

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next(
            (t for t in teleop if isinstance(t, KeyboardTeleop)), None
        )
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so100_leader.SO100Leader,
                        so101_leader.SO101Leader,
                        koch_leader.KochLeader,
                        ViserLeader,
                    ),
                )
            ),
            None,
        )

        if not (
            teleop_arm
            and teleop_keyboard
            and len(teleop) == 2
            and robot.name == "lekiwi_client"
        ):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix="observation"
            )

        # Get action from either policy or teleop
        if (
            policy is not None
            and preprocessor is not None
            and postprocessor is not None
        ):
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )

            action_names = dataset.features["action"]["names"]
            act_processed_policy: RobotAction = {
                f"{name}": float(action_values[i])
                for i, name in enumerate(action_names)
            }

        elif policy is None:
            act = teleop.get_action()

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Applies a pipeline to the action, default is IdentityProcessor
        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(
                dataset.features, action_values, prefix="action"
            )
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = (
        make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=cfg.dataset.video,
        ),
    )

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(
            dataset, robot, cfg.dataset.fps, dataset_features
        )
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
            * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    # Load pretrained policy
    policy = (
        None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    )
    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while (
            recorded_episodes < cfg.dataset.num_episodes
            and not events["stop_recording"]
        ):
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1)
                or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
