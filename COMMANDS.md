# Common Commands

Run these commands from the Isaac Lab root directory. This repository is assumed
to be checked out as `GO2W-HIM`.

## Train

```bash
TERM=xterm ./isaaclab.sh -p GO2W-HIM/scripts/train_him.py \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --video \
  --video_interval 6000 \
  --video_length 400 \
  --run_name baseline
```

## Resume Train

```bash
TERM=xterm ./isaaclab.sh -p GO2W-HIM/scripts/train_him.py \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --resume \
  --resume_path logs/him_rsl_rl/robotlab_go2w_him/<run>/model_200.pt \
  --video \
  --video_interval 6000 \
  --video_length 400 \
  --run_name baseline_resume
```

## Play

```bash
TERM=xterm ./isaaclab.sh -p GO2W-HIM/scripts/play_him.py \
  --device cuda:0 \
  --checkpoint logs/him_rsl_rl/robotlab_go2w_him/<run>/model_200.pt \
  --keyboard \
  --real_time
```

## Preview Terrain

```bash
TERM=xterm ./isaaclab.sh -p GO2W-HIM/scripts/preview_robotlab_terrain.py \
  --device cuda:0
```

## Optional Arguments

Training: `--task`, `--num_envs`, `--max_iterations`, `--num_steps_per_env`,
`--num_mini_batches`, `--seed`, `--run_name`, `--resume`, `--resume_path`,
`--video`, `--video_interval`, `--video_length`, `--profile_timing`.

Play: `--checkpoint`, `--num_envs`, `--device`, `--keyboard`, `--real_time`.

Terrain preview: `--device`, `--rows`, `--cols`, `--border-width`.

## Task IDs

```text
RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-v0
RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-Play-v0
```
