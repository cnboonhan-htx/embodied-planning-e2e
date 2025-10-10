from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.robot import Robot
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from utils_a2 import record_loop
from lerobot.processor.factory import make_default_processors
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from pathlib import Path
from utils_a2 import ViserFollower, ViserLeader, camera_config, TASK_DESCRIPTION, POLICY_REPO_NAME, DATA_REPO_NAME

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 120
RESET_TIME_SEC = 20

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
robot = ViserFollower(camera_config)
teleop = ViserLeader()
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

print("Action features:", action_features)
print("Observation features:", obs_features)
print("Dataset features keys:", list(dataset_features.keys()))

dataset_root = Path(f"/home/cnboonhan/data_collection/{DATA_REPO_NAME}").expanduser()
dataset_exists = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()
if dataset_exists:
    # Case 1: Dataset already exists on disk - load it
    print("✅ Found existing dataset on disk, loading...")
    dataset = LeRobotDataset(DATA_REPO_NAME, root=str(dataset_root))
    print("✅ Loaded existing dataset, will append new episodes")
else:
    # Case 2: Dataset doesn't exist - create new one
    print("📝 No existing dataset found, creating new dataset...")
    dataset = LeRobotDataset.create(
        repo_id=DATA_REPO_NAME,
        root=str(dataset_root),
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )
    print("✅ Created new dataset")


_, events = init_keyboard_listener()
init_rerun(session_name="recording")
robot.connect()
teleop.connect()

episode_idx = 0
print("Episode index: ", episode_idx)
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    print(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()