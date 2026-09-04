# G1-4010 tracker local simulation

This setup uses the 1.501B-step tracker checkpoint and the four references in
`/mnt/real_robot_reference_candidates`. The generated ONNX assets are local
and are intentionally not committed.

## Assets

- Policy: `storage/policy/4010_tracker_jump_reward_1p501b/checkpoints/001501102080/policy.onnx`
- References:
  - `storage/data/rr_stand_still/ref_data.onnx`
  - `storage/data/rr_walk_slow/ref_data.onnx`
  - `storage/data/rr_two_foot_jump/ref_data.onnx`
  - `storage/data/rr_hurdle_jump/ref_data.onnx`

The source NPZ files are 150 Hz. The deployment references are resampled to
50 Hz and qvel is reconstructed with the same backward-difference convention
as tracker training:

```bash
/mnt/Humanoid_Pipeline/.venv/bin/python \
  scripts/process_motion/prepare_4010_reference.py \
  --input INPUT.npz \
  --output deploy/storage/data/MOTION/ref_data.onnx
```

Use `--transition-seconds 0` for the three moving candidates because they
already contain transitions from and back to the tracker default pose.

## Build

```bash
cd /mnt/OpenTrack/deploy
./build_w_torque_projection.sh
```

## Interactive simulation

Start the deployment controller first:

```bash
cd /mnt/OpenTrack/deploy
export G1_TRACKER_POLICY=4010_tracker_jump_reward_1p501b
export G1_TRACKER_MOTIONS=rr_stand_still,rr_walk_slow,rr_two_foot_jump,rr_hurdle_jump
./start_sim_w_torque_projection.sh --iface lo
```

Then start MuJoCo in another terminal:

```bash
cd /mnt/OpenTrack/deploy/sim_interface
export CYCLONEDDS_HOME=/mnt/cyclonedds-0.10/install
export LD_LIBRARY_PATH=/mnt/cyclonedds-0.10/install/lib
/mnt/Humanoid_Pipeline/.venv/bin/python main.py --iface lo
```

Keyboard sequence:

1. `4` (R2): enable the controller.
2. `a`: DAMPING to STAND.
3. `x`: STAND to LOCO.
4. `9`: enable simulation physics.
5. Arrow key: select the tracker reference.
6. `6` (F1): exit the deployment controller.

Reference slots:

- Up: stand still
- Down: slow walk
- Left: two-foot jump
- Right: hurdle jump

The default 1.0 s controller blend is retained. The tracker reference clock is
held at frame zero during this blend and starts advancing only after the blend
ends. `G1_TRACKER_BLEND_SECONDS` can override the duration for experiments,
but the default should be retained for robot preparation.

## Final headless smoke-test results

Torque projection was enabled in every run.

| Reference | Joint MAE | Root z peak actual / ref | Torque-projected steps | Result |
|---|---:|---:|---:|---|
| stand | 0.061 rad | 0.792 / 0.802 m | 0.0% | stable |
| walk | 0.051 rad | 0.816 / 0.820 m | 3.0% | stable, distance 1.554 / 1.570 m |
| two-foot jump | 0.080 rad | 0.966 / 1.243 m | 21.1% | stable, insufficient height |
| hurdle jump | 0.065 rad | 0.986 / 1.114 m | 22.9% | stable, forward/orientation overshoot |

All four runs completed, returned to LOCO automatically, and exited with code
zero. Standing and slow walking are suitable for the next staged simulation
checks. The two jumps are not approved for real-robot execution yet.

Note: the three moving source files start and end at the exact deployment
default joint pose. `stand_still_150hz.npz` does not: its maximum initial
joint difference is 0.495 rad, so do not treat it as a default-pose hold.
