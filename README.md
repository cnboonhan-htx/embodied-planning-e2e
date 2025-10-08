# embodied-ai-toolkit


## Install
```
git clone https://github.com/cnboonhan/embodied-ai-toolkit.git --recurse-submodules
cd embodied-ai-toolkit; uv venv; uv sync; cd ..
cd humanoid-joint-teleop; uv venv; uv sync; cd ..

# create lerobot: https://github.com/huggingface/lerobot
conda install ffmpeg=7.1.1 -c conda-forge
pip install 'lerobot[feetech]' 

# Copy ffmpeg to Orin
wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl.tar.xz
scp [file] agibot:~/ffmpeg-manual

# set up usb
# Copy 99-webcam.rules to /etc/udev/rules.d in orin
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
ssh-copy-id agi@192.168.2.50
ssh agi@192.168.2.50 -R 5000:127.0.0.1:5000 -R 5001:127.0.0.1:5001
aima em stop-app hal_sensor
ssh-copy-id agi@192.168.100.100
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

# Setup Camera
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=2 video_nr=8,9 exclusive_caps=1,1 card_label="HeadCam,WristCam"
sudo v4l2-ctl --list-devices
# Stream Head Camera
ssh agi@192.168.2.50 "/agibot/data/home/agi/ffmpeg-manual/ffmpeg-master-latest-linuxarm64-gpl/bin/ffmpeg -f v4l2 -input_format yuyv422 -s 640x480 -i /dev/video2 -vcodec libx264 -preset veryfast -crf 23 -tune zerolatency -f mpegts -" | ffmpeg -i - -vcodec rawvideo -s 640x480 -vf format=yuv420p -f v4l2 /dev/video8

# Stream Wrist Camera
ssh agi@192.168.2.50 "/agibot/data/home/agi/ffmpeg-manual/ffmpeg-master-latest-linuxarm64-gpl/bin/ffmpeg -f v4l2 -input_format yuyv422 -s 640x480 -i /dev/video10 -vcodec libx264 -preset veryfast -crf 23 -tune zerolatency -f mpegts -" | ffmpeg -i - -vcodec rawvideo -s 640x480 -vf format=yuv420p -f v4l2 /dev/video9

# Test Cameras
ffplay -f v4l2 /dev/video8
ffplay -f v4l2 /dev/video9

source ~/miniconda3/bin/activate; conda activate lerobot
python3 src/record_a2.py

# Inspect dataset: https://huggingface.co/spaces/lerobot/visualize_dataset
# Train dataset

lerobot-train --dataset.repo_id=cnboonhan-htx/a2-pnp-0810-right-hand-lift --policy.type=act --output_dir=outputs/train/cnboonhan-htx/a2-pnp-0810-right-hand-pet --job_name=a2-pnp-0810-right-hand-pet --policy.device=cuda --wandb.enable=false --policy.repo_id=cnboonhan-htx/a2-pnp-0810-right-hand-pet

# Inference
source ~/miniconda3/bin/activate; conda activate lerobot
python3 src/inference_a2.py
```