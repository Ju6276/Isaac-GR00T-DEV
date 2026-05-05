# GR00T N1.7 + RoboCasa GR1 Pipeline

这份文档记录当前仓库如何把 **GR00T N1.7** 接到 **RoboCasa GR1 tabletop**：训练、24 任务 co-training、仿真评估、数据检查，以及每个代码改动的作用。

当前分支建议：`robocasa`  
当前基础 commit：`23ace64 GR00T N1.7 Release`

---

## 1. 集成关系

我们不是把 RoboCasa 环境直接合进 GR00T，而是做了两层对接：

```text
训练:
RoboCasa GR1 LeRobot 数据集
  -> GR00T dataset loader
  -> NEW_EMBODIMENT modality config
  -> fine-tuned GR00T checkpoint

评估:
RoboCasa env
  -> scripts/eval/eval_robocasa_gr1_gr00t.py adapter
  -> GR00T PolicyClient
  -> gr00t/eval/run_gr00t_server.py
  -> fine-tuned GR00T checkpoint
  -> action 回到 RoboCasa env
```

现在仍然复用 DiT4DiT repo 的两部分：

- 数据目录：`/home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`
- RoboCasa rollout harness：`examples.Robocasa_tabletop.eval_files.simulation_env`

不再需要 DiT4DiT policy server 或 DiT4DiT checkpoint。

---

## 2. 代码改动

本仓库为了跑通 RoboCasa GR1 做了 5 处代码适配：

| 文件 | 作用 |
|---|---|
| `examples/robocasa_gr1/gr1_config.py` | 注册 RoboCasa GR1 的 `NEW_EMBODIMENT` modality config |
| `gr00t/configs/finetune_config.py` | 给训练 CLI 增加 `--dataset-paths`，支持多任务 co-training |
| `gr00t/experiment/launch_finetune.py` | 加载自定义 modality config、接收多个 dataset、启用本地 Cosmos 和 sin/cos state |
| `scripts/eval/eval_robocasa_gr1_gr00t.py` | RoboCasa 仿真环境到 GR00T policy server 的 adapter |
| `scripts/eval/check_robocasa_gr1_dataset.py` | 检查 24 个 RoboCasa GR1 数据集的 `meta/data/videos` 是否齐全 |

### 2.1 注册 RoboCasa GR1

文件：`examples/robocasa_gr1/gr1_config.py`

GR00T N1.7 没有原生 `GR1` embodiment tag，所以用官方预留的：

```text
NEW_EMBODIMENT
```

注册代码：

```python
register_modality_config(gr1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

当前配置对齐 DiT4DiT 的 `FourierGr1ArmsWaistDataConfig`：

```text
state/action order:
left_arm, right_arm, left_hand, right_hand, waist

dims:
left_arm 7
right_arm 7
left_hand 6
right_hand 6
waist 3
total 29
```

训练使用的 modality：

```text
video:    ego_view
state:    left_arm, right_arm, left_hand, right_hand, waist
action:   left_arm, right_arm, left_hand, right_hand, waist
language: annotation.human.coarse_action
```

state 使用 sin/cos：

```python
"state": ModalityConfig(
    delta_indices=[0],
    modality_keys=GR1_STATE_KEYS,
    sin_cos_embedding_keys=GR1_STATE_KEYS,
)
```

action 是 16 步 absolute joint target：

```python
"action": ModalityConfig(
    delta_indices=list(range(0, 16)),
    modality_keys=GR1_ACTION_KEYS,
    action_configs=[
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        )
    ] * len(GR1_ACTION_KEYS),
)
```

### 2.2 支持多数据集 co-training

文件：`gr00t/configs/finetune_config.py`

新增：

```python
dataset_path: str | None = None
dataset_paths: list[str] | None = None
```

文件：`gr00t/experiment/launch_finetune.py`

训练时会合并单数据集和多数据集入口：

```python
dataset_paths = ft_config.dataset_paths or (
    [ft_config.dataset_path] if ft_config.dataset_path is not None else []
)
```

因此可以用：

```bash
--dataset-path /path/to/one/task
```

或者：

```bash
--dataset-paths "${DATASET_PATHS[@]}"
```

### 2.3 N1.7 本地 Cosmos 路径和动作配置

文件：`gr00t/experiment/launch_finetune.py`

当前覆盖：

```python
config.model.model_name = "/home/d024/models/nvidia/Cosmos-Reason2-2B"
config.model.backbone_trainable_params_fp32 = True
config.model.apply_sincos_state_encoding = True
config.model.use_relative_action = False
```

本地 Cosmos 路径必须包含 `nvidia/Cosmos-Reason2` 字串，因为 `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` 会用字符串判断 backbone 类型。

### 2.4 RoboCasa 评估 adapter

文件：`scripts/eval/eval_robocasa_gr1_gr00t.py`

作用：

- 复用 DiT4DiT 的 `run_evaluation`
- 从 RoboCasa observation 取 `video.ego_view` 和 5 组 state
- 调用 GR00T `PolicyClient`
- 把 GR00T 输出的 5 组 action 还给 RoboCasa env

adapter 不手动做 sin/cos；sin/cos 由微调 checkpoint 的 processor 配置处理。

---

## 3. 环境和资产

### 3.1 GR00T 训练环境

```bash
cd /home/d024/Isaac-GR00T
source .venv/bin/activate
which python
```

期望 Python 来自：

```text
/home/d024/Isaac-GR00T/.venv/bin/python
```

### 3.2 模型资产

```text
GR00T-N1.7-3B:
/home/d024/models/GR00T-N1.7-3B

Cosmos-Reason2-2B:
/home/d024/models/Cosmos-Reason2-2B

Cosmos symlink:
/home/d024/models/nvidia/Cosmos-Reason2-2B -> /home/d024/models/Cosmos-Reason2-2B
```

下载示例：

```bash
huggingface-cli download nvidia/GR00T-N1.7-3B \
  --local-dir /home/d024/models/GR00T-N1.7-3B

huggingface-cli download nvidia/Cosmos-Reason2-2B \
  --local-dir /home/d024/models/Cosmos-Reason2-2B

mkdir -p /home/d024/models/nvidia
ln -sfn /home/d024/models/Cosmos-Reason2-2B /home/d024/models/nvidia/Cosmos-Reason2-2B
```

两个模型都是 gated repo，需要先在 Hugging Face 网页 accept license，再 `huggingface-cli login`。

### 3.3 RoboCasa/DiT4DiT 环境

训练在 GR00T `.venv` 中跑；仿真评估在 conda `robocasa` 环境中跑：

```bash
cd /home/d024/DiT4DiT
deactivate 2>/dev/null; true
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate robocasa
export PYTHONPATH=/home/d024/DiT4DiT:/home/d024/Isaac-GR00T:$PYTHONPATH
```

已修复过的依赖：

```bash
/home/d024/miniconda3/envs/robocasa/bin/python -m pip install "numpy==1.26.4"
/home/d024/miniconda3/envs/robocasa/bin/python -m pip install "websockets>=13,<16"
```

确认当前环境实际 import 的 RoboCasa：

```bash
python - <<'PY'
import robocasa
print(robocasa.__file__)
PY
```

期望路径类似：

```text
/home/d024/robocasa-gr1-tabletop-tasks/robocasa/__init__.py
```

---

## 4. 数据集

数据根目录：

```text
/home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
```

单任务数据：

```text
gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000
```

24 任务 co-training 使用：

```text
gr1_unified.*GR1ArmsAndWaistFourierHands_1000
```

DiT4DiT 对应配置：

```text
data_mix: fourier_gr1_unified_1000
mixture file: /home/d024/DiT4DiT/DiT4DiT/dataloader/gr00t_lerobot/mixtures.py
tasks: 24
weight per task: 1.0
max_train_steps: 200000
state/action: left_arm, right_arm, left_hand, right_hand, waist
```

检查数据完整性：

```bash
cd /home/d024/Isaac-GR00T
python scripts/eval/check_robocasa_gr1_dataset.py
```

成功输出：

```text
OK: all 24 datasets are complete (24000 episodes).
```

这个脚本检查：

- 24 个任务目录
- 每个任务 `meta/episodes.jsonl` 1000 行
- 每个任务 `data/chunk-000/episode_*.parquet` 1000 个
- 每个任务 `videos/chunk-000/observation.images.ego_view/episode_*.mp4` 1000 个
- 没有 0 字节空文件

---

## 5. 训练

### 5.1 单任务烟雾测试

```bash
cd /home/d024/Isaac-GR00T
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path /home/d024/models/GR00T-N1.7-3B \
  --dataset-path /home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/robocasa_gr1/gr1_config.py \
  --num-gpus 1 \
  --output-dir /home/d024/models/gr00t_n17_gr1_finetune \
  --experiment-name gr1_pnp_cup_to_drawer_dit4dit_state_smoke \
  --max-steps 20 \
  --global-batch-size 8 \
  --dataloader-num-workers 4 \
  --save-steps 20 \
  --use-wandb \
  --wandb-project finetune-gr00t-n1d7-gr1
```

### 5.2 单任务正式训练

```bash
cd /home/d024/Isaac-GR00T
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path /home/d024/models/GR00T-N1.7-3B \
  --dataset-path /home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/robocasa_gr1/gr1_config.py \
  --num-gpus 1 \
  --output-dir /home/d024/models/gr00t_n17_gr1_finetune \
  --experiment-name gr1_pnp_cup_to_drawer_dit4dit_state \
  --max-steps 30000 \
  --global-batch-size 8 \
  --dataloader-num-workers 4 \
  --save-steps 5000 \
  --save-total-limit 3 \
  --use-wandb \
  --wandb-project finetune-gr00t-n1d7-gr1
```

已跑通结果：

```text
experiment: /home/d024/models/gr00t_n17_gr1_finetune/gr1_pnp_cup_to_drawer_dit4dit_state
max_steps: 30000
runtime: 2:24:57
train_loss: 0.09864946630746126
last logged loss at step 30000: 0.0309
```

### 5.3 24 任务 co-training

```bash
cd /home/d024/Isaac-GR00T
source .venv/bin/activate

DATA_ROOT=/home/d024/DiT4DiT/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim

mapfile -t DATASET_PATHS < <(
  find "$DATA_ROOT" -maxdepth 1 -type d \
    -name 'gr1_unified.*GR1ArmsAndWaistFourierHands_1000' | sort
)

printf "Using %d datasets\n" "${#DATASET_PATHS[@]}"
printf '%s\n' "${DATASET_PATHS[@]}"

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path /home/d024/models/GR00T-N1.7-3B \
  --dataset-paths "${DATASET_PATHS[@]}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/robocasa_gr1/gr1_config.py \
  --num-gpus 1 \
  --output-dir /home/d024/models/gr00t_n17_gr1_finetune \
  --experiment-name gr1_unified_24tasks_dit4dit_state \
  --max-steps 200000 \
  --global-batch-size 8 \
  --dataloader-num-workers 4 \
  --save-steps 25000 \
  --save-total-limit 4 \
  --use-wandb \
  --wandb-project finetune-gr00t-n1d7-gr1
```

确认看到：

```text
Using 24 datasets
```

输出目录：

```text
/home/d024/models/gr00t_n17_gr1_finetune/gr1_unified_24tasks_dit4dit_state
```

中间 checkpoint：

```text
checkpoint-25000
checkpoint-50000
...
checkpoint-200000
```

推理时可以用实验目录顶层，也可以用某个 checkpoint：

```text
/home/d024/models/gr00t_n17_gr1_finetune/gr1_unified_24tasks_dit4dit_state
/home/d024/models/gr00t_n17_gr1_finetune/gr1_unified_24tasks_dit4dit_state/checkpoint-200000
```

### 5.4 默认训练模块

| 模块 | 默认是否训练 |
|---|---|
| Cosmos-Reason2 LLM | 否 |
| Cosmos-Reason2 visual encoder | 否 |
| multimodal projector | 是 |
| diffusion action head | 是 |

也就是冻结 VLM 主干，只训练 projector 和 action head。

---

## 6. 推理评估

### 6.1 启动 GR00T policy server

注意：`NEW_EMBODIMENT` 不能直接用于 base model。必须使用微调后的 checkpoint，因为 processor 里才有 RoboCasa GR1 的 modality config。

24 任务模型：

```bash
cd /home/d024/Isaac-GR00T
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
  --model-path /home/d024/models/gr00t_n17_gr1_finetune/gr1_unified_24tasks_dit4dit_state \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 6398 \
  --no-strict \
  --use-sim-policy-wrapper
```

单任务模型：

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
  --model-path /home/d024/models/gr00t_n17_gr1_finetune/gr1_pnp_cup_to_drawer_dit4dit_state \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 6398 \
  --no-strict \
  --use-sim-policy-wrapper
```

参数说明：

- `--use-sim-policy-wrapper`：接受 `video.ego_view`、`state.left_arm` 这类 flat key，并转换到 GR00T 内部格式。
- `--no-strict`：关闭严格 key/shape 校验，外接仿真环境时更稳。

### 6.2 启动 RoboCasa 评估

```bash
cd /home/d024/DiT4DiT
deactivate 2>/dev/null; true
source /home/d024/miniconda3/etc/profile.d/conda.sh
conda activate robocasa

export PYTHONPATH=/home/d024/DiT4DiT:/home/d024/Isaac-GR00T:$PYTHONPATH

/home/d024/miniconda3/envs/robocasa/bin/python \
  /home/d024/Isaac-GR00T/scripts/eval/eval_robocasa_gr1_gr00t.py \
  --args.host 127.0.0.1 \
  --args.port 6398 \
  --args.env-name "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env" \
  --args.n-episodes 5 \
  --args.n-envs 1 \
  --args.max-episode-steps 720 \
  --args.n-action-steps 12
```

视频默认保存到 adapter 默认目录。也可以显式指定：

```bash
--args.video-out-path /home/d024/models/gr00t_n17_gr1_finetune/eval_videos
```

文件名里：

```text
success1 = 成功 episode
success0 = 失败 episode
```

### 6.3 检查进程和视频

```bash
ps -ef | rg "run_gr00t_server|eval_robocasa_gr1_gr00t|simulation_env|6398"
ss -ltnp | rg ":6398"
```

```bash
find /home/d024/models/gr00t_n17_gr1_finetune \
  -type f -name "*.mp4" \
  -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" | sort | tail -30
```

```bash
ffprobe -v error \
  -show_entries format=duration,size \
  -show_entries stream=width,height,nb_frames,r_frame_rate \
  -of default=noprint_wrappers=1 \
  /path/to/video.mp4
```

单任务 30k checkpoint 已有一次 5 episode 评估：

```text
success: 1 / 5 = 20%
video: 1280x800, 10 fps, 360 frames, 36s
```

---

## 7. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `Embodiment tag NEW_EMBODIMENT is not supported by this checkpoint` | 用 base model 启动了 `NEW_EMBODIMENT` | 使用微调后的模型目录或 checkpoint |
| `GatedRepoError ... Cosmos-Reason2-2B` | 还在访问 HF gated repo | 用 `/home/d024/models/nvidia/Cosmos-Reason2-2B` 本地路径并登录 HF |
| `Unsupported model name: /xxx/Cosmos-Reason2-2B` | 路径缺少 `nvidia/Cosmos-Reason2` 字串 | 使用当前 symlink 路径 |
| `AssertionError: numpy version must be...` | RoboCasa fork 不支持 NumPy 2.x | 在 robocasa env 安装 `numpy==1.26.4` |
| `ModuleNotFoundError: websockets` | DiT4DiT eval 需要 websockets | 安装 `websockets>=13,<16` |
| 训练读 parquet/mp4 报 `FileNotFoundError` | 数据集不完整 | 跑 `scripts/eval/check_robocasa_gr1_dataset.py` 并补齐缺文件 |
| 手往下但抓不起杯子 | 数据/动作配置或训练不足 | 优先使用 DiT4DiT-style 5 组 state/action + sin/cos，增加训练步数或 co-training |

---
