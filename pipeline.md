git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

git checkout 4e62473d5226c55784697944a5c9606a51927bfc
git submodule update --init --recursive

uv sync
uv pip install -e .

source .venv/bin/activate

hf download  nvidia/GR00T-N1.6-3B  --local-dir GR00T-N1.6-3B
huggingface-cli download nvidia/GR00T-N1.6-G1-PnPAppleToPlate --local-dir ./GR00T-N1.6-G1-PnPAppleToPlate


#
source .venv/bin/activate
cd /home/d024/Isaac-GR00T
.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment_tag UNITREE_G1 \
    --use_sim_policy_wrapper

 
# 本地路径
.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path ./GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment_tag UNITREE_G1 \
    --use_sim_policy_wrapper


# Terminal 1 - 先启动服务器：
source .venv/bin/activate
cd /home/d024/Isaac-GR00T

uv run --no-sync python gr00t/eval/run_gr00t_server.py \
    --model-path ./GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper


# Terminal 2 - 运行客户端评估：
cd /home/d024/Isaac-GR00T

gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --max_episode_steps=1440 \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --n_action_steps 20 \
    --n_envs 1 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555