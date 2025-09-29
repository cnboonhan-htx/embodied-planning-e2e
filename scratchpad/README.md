# README


## Install
```
git clone [repo] --recurse-submodules
cd embodied-ai-toolkit; uv venv; [install]
cd humanoid-joint-teleop; uv venv; [install]

# create lerobot: https://github.com/huggingface/lerobot
pip install 'lerobot[feetech]' 
```

## Run SO100 Collection
```
source embodied-ai-toolkit/.venv/bin/activate
uv run embodied-ai-toolkit/main.py  --config_path config.so101.json 

source humanoid-joint-teleop/.venv/bin/activate
uv run teleop_arm_so100.py 

source ~/miniconda3/bin/activate; conda activate lerobot
cd lerobot
python3 ../scratchpad/viser_record_so100.py
```

## Run A2 Collection
```
# Open 8 terminals, each paragraph is a new terminal

source embodied-ai-toolkit/.venv/bin/activate
uv run embodied-ai-toolkit/main.py  --config_path action_config.a2.json 
# Check 127.0.0.1:8080

source embodied-ai-toolkit/.venv/bin/activate
uv run embodied-ai-toolkit/main.py  --config_path state_config.a2.json 
# Check 127.0.0.1:8081

source humanoid-joint-teleop/.venv/bin/activate
uv run humanoid-joint-teleop/teleop_arm.py 
# Check 127.0.0.1:8080 and move the teleop 

# password is 1
ssh agi@192.168.2.50 -R 5000:127.0.0.1:5000 -R 5001:127.0.0.1:5001
x86
aima em stop-app motion_player
aima em load-env
python3 ~/bh-new-teleop/ros2_state_publisher.py
# Check 127.0.0.1:8081 that the state is updated

# password is 1
ssh agi@192.168.2.50
x86
aima em load-env
## HAND WILL MOVE SLOWLY
grpcurl -format json -plaintext -d '{"update_frequency": 100.0}' localhost:5000 rosbot_api.RobotApiService/StreamJointData | python3 ~/bh-new-teleop/ros2_streamer_slow.py 
[Ctrl-C]
## HAND WILL MOVE QUICKLY
grpcurl -format json -plaintext -d '{"update_frequency": 100.0}' localhost:5000 rosbot_api.RobotApiService/StreamJointData | python3 ~/bh-new-teleop/ros2_streamer.py 
# Start moving teleop!

source ~/miniconda3/bin/activate; conda activate lerobot
cd scratchpad
python3 viser_record_a2.py

# Inspect dataset: https://huggingface.co/spaces/lerobot/visualize_dataset
# Train dataset
# TODO: Add command


# Inference
```