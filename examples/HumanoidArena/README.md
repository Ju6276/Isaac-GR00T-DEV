# GR00T N1.7 on HumanoidArena (SONIC-only, canonical 40D)

This integration trains GR00T N1.7 on only the SONIC subset of HumanoidArena. HumanoidArena is
kept as the simulator and benchmark evaluator. GR00T predicts the same canonical 40D reference-pose
action used by the published ACT, Diffusion Policy, Flow Matching, and pi0.5 baselines. It does not
predict SONIC's native 64D latent.

## Frozen interface

State (64D):

```text
[0:6]   root heading-canonical rotation 6D
[6:35]  canonical Unitree G1 joint positions (29)
[35:64] canonical Unitree G1 joint velocities (29)
```

Action (40D):

```text
[0:2]   reference-root base-local XY delta
[2:3]   reference-root Z
[3:9]   reference-root rotation 6D in the episode reference frame
[9:38]  canonical Unitree G1 joint reference positions (29)
[38:40] left/right binary hand commands
```

Every action component is configured as `ABSOLUTE`: the dataset already stores the benchmark's
reference-motion target. GR00T must not turn it into an observation-relative residual.

The action supervision horizon is 30 frames for the GR00T experiment. The OpenDoor recipe uses
100,000 optimizer updates and global batch size 8 to match the pi0.5 training budget. W&B is forced
to online mode under the default `ju-dong6276-technical-university-of-munich` entity. Horizon changes
only the number of future steps; every deployed step remains the exact HumanoidArena canonical 40D
action.

## Prepare data

First obtain or convert the SONIC-only HumanoidArena V3.1 LeRobot dataset. Then copy it into the
GR00T workspace and convert it to GR00T's LeRobot V2.1 format:

```bash
cd /path/to/Isaac-GR00T
uv run python examples/HumanoidArena/prepare_dataset.py \
  --input /path/to/HumanoidArenaV3.1/sonic_refpose_v3_1 \
  --output datasets/humanoidarena_sonic_v31
```

The preparation step rejects TWIST2 or mixed datasets and validates the exact 64D/40D schema.

Generate GR00T statistics after preparation:

```bash
uv run python gr00t/data/stats.py \
  --dataset-path datasets/humanoidarena_sonic_v31 \
  --embodiment-tag HUMANOIDARENA_G1 \
  --modality-config-path examples/HumanoidArena/humanoidarena_config.py
```

## Fine-tune

```bash
NUM_GPUS=1 MAX_STEPS=100000 GLOBAL_BATCH_SIZE=8 \
uv run bash examples/finetune.sh \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --dataset-path datasets/humanoidarena_sonic_v31 \
  --embodiment-tag HUMANOIDARENA_G1 \
  --modality-config-path examples/HumanoidArena/humanoidarena_config.py \
  --output-dir outputs/humanoidarena_sonic_groot_n17 \
  --experiment-name humanoidarena-sonic-canonical40
```

For the closest comparison with HumanoidArena's per-task baselines, prepare and train one dataset per
task. A combined seven-task model is a separate `SONIC x7` experiment.

For the fixed OpenDoor comparison, use the reproducible wrapper:

```bash
uv run bash examples/HumanoidArena/train_opendoor_sonic.sh
```

`BASE_MODEL_PATH` defaults to `GR00T-N1.7-3B` in the repository root and can
be overridden. The other six per-task recipes use the same defaults:

```bash
BASE_MODEL_PATH=/path/to/GR00T-N1.7-3B \
  uv run bash examples/HumanoidArena/train_sonic_task.sh football
```

## Closed-loop evaluation

Start the normal GR00T server with the trained checkpoint:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path outputs/humanoidarena_sonic_groot_n17/checkpoint-100000 \
  --embodiment-tag HUMANOIDARENA_G1 \
  --device cuda:0 --host 127.0.0.1 --port 5555
```

In a second GR00T shell, expose the HTTP protocol already consumed by HumanoidArena:

```bash
uv run python examples/HumanoidArena/serve_http_bridge.py \
  --gr00t-host 127.0.0.1 --gr00t-port 5555 \
  --http-host 127.0.0.1 --http-port 18080
```

Then run HumanoidArena's existing SONIC evaluation with:

```text
--input_source vla
--gmt_backend sonic
--lerobot_server_url http://127.0.0.1:18080
SONIC_VLA_ACTION_FORMAT=semantic_v3
```

Do not set `SONIC_VLA_ACTION_FORMAT=latent64` for this benchmark.

The companion evaluator changes live in
[`Ju6276/HumanoidArena-DEV`](https://github.com/Ju6276/HumanoidArena-DEV/tree/feat/humanoidarena-vla-adapters).
To train and evaluate the remaining six tasks sequentially, clone that repository
next to this one (or set `ARENA_ROOT`) and run:

```bash
export ARENA_ROOT=/path/to/HumanoidArena
export EVAL_PYTHON=/path/to/isaaclab/python
export SONIC_POLICY_ROOT=/path/to/sonic/wbc/policy/release
bash examples/HumanoidArena/train_eval_sonic_queue.sh
```

Each task is trained for 100,000 optimizer steps and then evaluated for 60
episodes (three seeds, 20 repeats per seed) before the next task starts.
