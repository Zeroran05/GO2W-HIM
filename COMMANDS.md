# Common Commands

Run the commands below from the repository root in an activated Isaac Lab environment.

## Install

Install the extension in editable mode:

```bash
python -m pip install -e source/robotlab_go2w_him
```

The training and play scripts load the bundled `rsl_rl` package directly from this repository.

## Train

Start the default RobotLab-based HIM training job:

```bash
python scripts/train_him.py --headless --device cuda:0
```

Specify the environment count, rollout length, iterations, seed, and run name:

```bash
python scripts/train_him.py \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
    --video \
  --video_interval 2000 \
  --video_length 200 \
  --run_name baseline
```

Run a short smoke test before a full training job:

```bash
python scripts/train_him.py \
  --headless \
  --device cuda:0 \
  --num_envs 64 \
  --num_steps_per_env 24 \
  --max_iterations 2 \
  --run_name smoke_test
```

## Resume

Resume from a specific checkpoint. Both options are required:

```bash
python scripts/train_him.py \
  --headless \
  --device cuda:0 \
  --resume \
  --resume_path logs/him_rsl_rl/robotlab_go2w_him/<run>/model_200.pt
```

## Record Training Video

```bash
python scripts/train_him.py \
  --headless \
  --device cuda:0 \
  --video \
  --video_interval 2000 \
  --video_length 200
```

`--video_interval` and `--video_length` use environment steps. Camera rendering reduces collection throughput.

## Profile Collection Time

Print one synchronized module timing sample per training iteration:

```bash
python scripts/train_him.py \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --num_steps_per_env 24 \
  --profile_timing
```

The timing report separates the HIM policy, physics decimation, rewards, terminations, reset processing, observations, and runner post-processing. It synchronizes the GPU on the sampled step, so remove `--profile_timing` for normal training benchmarks.

To compare this repository with RobotLab or `HIMLoco-Go2W-IsaacLab`, keep the environment count, rollout length, device, headless mode, and video settings identical.

## Play

Run a trained policy with sampled velocity commands:

```bash
python scripts/play_him.py \
  --device cuda:0 \
  --checkpoint logs/him_rsl_rl/robotlab_go2w_him/<run>/model_200.pt
```

Run one robot with keyboard velocity control and real-time pacing:

```bash
python scripts/play_him.py \
  --device cuda:0 \
  --checkpoint logs/him_rsl_rl/robotlab_go2w_him/<run>/model_200.pt \
  --keyboard \
  --real_time
```

## Preview Terrain

Open the complete 10-by-20 RobotLab training terrain without spawning robots:

```bash
python scripts/preview_robotlab_terrain.py --device cuda:0
```

Use a smaller grid when only checking terrain generation and rendering:

```bash
python scripts/preview_robotlab_terrain.py \
  --device cuda:0 \
  --rows 5 \
  --cols 10 \
  --border-width 10
```

## Logs and Checkpoints

Open TensorBoard:

```bash
tensorboard --logdir logs/him_rsl_rl/robotlab_go2w_him --port 6006
```

List recently modified checkpoints:

```bash
find logs/him_rsl_rl/robotlab_go2w_him -name 'model_*.pt' -print0 \
  | xargs -0 ls -lt \
  | head
```

## Task IDs

```text
RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-v0
RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-Play-v0
```
