# embodied-ai-toolkit


## Install
```
git clone https://github.com/cnboonhan/embodied-ai-toolkit.git --recurse-submodules
cd embodied-ai-toolkit; uv venv; uv sync; cd ..
cd humanoid-joint-teleop; uv venv; uv sync; cd ..

# create lerobot: https://github.com/huggingface/lerobot
pip install 'lerobot[feetech]' 
```

## Run A2 Collection
```
# Open 8 terminals, each paragraph is a new terminal

source embodied-ai-toolkit/.venv/bin/activate
uv run embodied-ai-toolkit/main.py  --config_path config/action_config.a2.json 
# Check 127.0.0.1:8080

source embodied-ai-toolkit/.venv/bin/activate
uv run embodied-ai-toolkit/main.py  --config_path config/state_config.a2.json 
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
python3 src/record_a2.py

# Inspect dataset: https://huggingface.co/spaces/lerobot/visualize_dataset
# Train dataset
lerobot-train --dataset.repo_id=cnboonhan-htx/a2-pnp-3009-right-hand --policy.type=diffusion --output_dir=outputs/train/cnboonhan-htx/a2-pnp-3009-right-hand --job_name=a2-pip-3009-right-hand --policy.device=cuda --wandb.enable=true --policy.repo_id=cnboonhan-htx/a2-pnp-3009-right-hand

# Inference
python3 src/inference_a2.py
```