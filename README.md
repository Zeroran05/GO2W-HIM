# RobotLab Go2W HIM IsaacLab

Independent IsaacLab extension combining the RobotLab Unitree Go2W task baseline with the HIMLoco representation-learning policy.

## Design

- RobotLab Go2W URDF, rough terrain, rewards, events and curriculum.
- RobotLab implicit actuators (`Kp=25`, `Kd=0.5`) with no action delay.
- HIMLoco actor contract: 57 values per step and six-frame history (`342` values).
- HIMLoco critic contract: `262` privileged values.
- HIM estimator output: 3-D velocity estimate and 16-D latent vector.
- 16-D action order: `FL`, `FR`, `RL`, `RR`, with each leg ordered as hip, thigh, calf, wheel.

The extension bundles the Go2W URDF and meshes and does not import the RobotLab Python package.

## Install

From an IsaacLab checkout:

```bash
./isaaclab.sh -p -m pip install -e /path/to/RobotLab-Go2W-HIM-IsaacLab/source/robotlab_go2w_him
```

The training and play scripts prepend the bundled `rsl_rl/` directory automatically, so it does not need to replace IsaacLab's installed RSL-RL package.

## Train

```bash
./isaaclab.sh -p /path/to/RobotLab-Go2W-HIM-IsaacLab/scripts/train_him.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-v0 \
  --num_envs 4096 \
  --device cuda:0 \
  --headless
```

The default rollout length is 24 steps. Add `--video` for periodic training videos or `--profile_timing` for one synchronized module-timing sample per iteration.

## Play

```bash
./isaaclab.sh -p /path/to/RobotLab-Go2W-HIM-IsaacLab/scripts/play_him.py \
  --checkpoint /path/to/model_1000.pt \
  --device cuda:0
```

Keyboard control:

```bash
./isaaclab.sh -p /path/to/RobotLab-Go2W-HIM-IsaacLab/scripts/play_him.py \
  --checkpoint /path/to/model_1000.pt \
  --device cuda:0 \
  --keyboard \
  --real_time
```

Arrow up/down command forward/backward velocity, arrow left/right command lateral velocity, `Z`/`X` command yaw, and `L` clears the command.

## Tasks

- `RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-v0`
- `RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-HIM-Play-v0`

## Attribution

The environment configuration and MDP terms are derived from RobotLab. The vendored HIM algorithm is derived from HIMLoco and its RSL-RL fork. Their original license headers and the vendored RSL-RL license are retained.
