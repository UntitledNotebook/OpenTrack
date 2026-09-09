# ScaleTrack G1-4010 deployment

This path is independent of the legacy `FsmTrackerController` and the VAE
controller.  ScaleTrack's medium Transformer uses a different interface:

- 3 frames of root quaternion, body-frame angular velocity, joint position,
  and joint velocity;
- 3 frames of the policy's normalized action;
- 6 future samples (`[0,1,2,3,4,5]`) of 14 selected rigid-body poses;
- one control-mode index (`7`, `WholeBody-14`, by default).

The exported `policy.onnx` contains ScaleTrack's proprioception assembly,
G1 forward kinematics, mode masking, Transformer actor, action scaling, and
default-position offset.  Its `motor_targets` output is already an absolute
29-DoF position target.

## 1. Export a checkpoint

Use the ScaleBFM environment because it contains the training PyTorch build:

```bash
cd /mnt/OpenTrack
/mnt/ScaleBFM/.venv/bin/python scripts/export_scaletrack_policy.py \
  --checkpoint /mnt/ScaleBFM/ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/g1_4010_m_pretrained_landingdr_v1_seed42 \
  --output-root deploy/storage/policy/scaletrack_4010_landingdr_v1
```

Passing the run directory selects the largest `model_<iteration>.pt`.  The
exporter uses `torch.load(..., weights_only=True)` and writes:

```text
deploy/storage/policy/scaletrack_4010_landingdr_v1/checkpoints/<iteration>/
  policy.onnx
  metadata.json
```

The metadata is not decorative: the C++ controller rejects a wrong joint
order, context length, frequency, or future-offset layout and loads the exact
ScaleTrack stiffness/damping values from it.

## 2. Prepare a reference

For a packaged ScaleTrack motion containing `body_pos_w`, `body_quat_w`, and
`fps`:

```bash
/mnt/Humanoid_Pipeline/.venv/bin/python \
  scripts/process_motion/prepare_scaletrack_reference.py \
  --input /path/to/processed_motion.npz \
  --output deploy/storage/data/my_motion/scaletrack_ref.onnx
```

The same command also accepts a MuJoCo-style NPZ containing `qpos [N,36]` and
`frequency`.  In that case it runs FK with OpenTrack's G1-4010 scene.  A
missing frequency can be supplied with `--source-frequency`; input is
resampled onto an exact 50 Hz clock.

The reference stores the 14 body poses in this fixed order:

```text
pelvis,
left hip-roll / knee / ankle-roll,
right hip-roll / knee / ankle-roll,
torso,
left shoulder-roll / elbow / wrist-yaw,
right shoulder-roll / elbow / wrist-yaw
```

OpenTrack has no global root-position estimate on the robot.  The controller
therefore follows ScaleTrack's local-tracking convention: current reference
root translation is used as the virtual robot translation, initial yaw is
aligned to the measured IMU yaw, and measured roll/pitch/yaw is retained when
forming all body targets.

## 3. Run in MuJoCo

Terminal A starts the virtual robot (a headless example is shown):

```bash
cd /mnt/OpenTrack/deploy/sim_interface
/mnt/Humanoid_Pipeline/.venv/bin/python main.py --iface lo --no-viewer
```

Terminal B builds and starts the ScaleTrack registry with torque projection:

```bash
cd /mnt/OpenTrack/deploy
./build_w_torque_projection.sh
./start_scaletrack_sim.sh scaletrack_4010_landingdr_v1 my_motion --iface lo
```

Use the normal FSM sequence: `R2 -> A -> X -> SimStart -> D-pad Up`.  Multiple
comma-separated motions occupy slots in order.  `G1_SCALETRACK_MODE` may select
mode `0..7`; leave it at `7` for full-body tracking.  The reference clock is
held at frame zero during the DANCE transition blend, just like the other
deployment controllers.

## 4. Run on a real G1-4010

Do not start with the robot standing freely. Suspend it securely for the first
low-level test, keep the emergency stop accessible, and validate every motion
in MuJoCo first. On the robot, identify the Ethernet interface connected to
the G1 DDS network (commonly `eth0`) and run the asset-only preflight:

```bash
cd /path/to/OpenTrack/deploy
./build_w_torque_projection.sh
./start_scaletrack_deploy.sh   scaletrack_4010_landingdr_v1   scaletrack_single_jump,scaletrack_walk_slow   --iface eth0   --check-only
```

After the preflight reports `PASS`, remove `--check-only`:

```bash
./start_scaletrack_deploy.sh   scaletrack_4010_landingdr_v1   scaletrack_single_jump,scaletrack_walk_slow   --iface eth0
```

The real-robot launcher always uses the torque-projection build and refuses
`lo`, a missing interface, incomplete policy/reference assets, an invalid
ScaleTrack mode, or a `CYCLONEDDS_URI` restricted to loopback. It prints the
selected numeric checkpoint before starting; the controller then waits at
`Press R2 to start!`. Use the Unitree remote in the normal sequence:
`R2 -> A -> X -> D-pad`. There is no `SimStart` step on hardware.

## 5. Current smoke-test status

The local `model_400.pt` export was checked with ONNX Runtime on CPU:

- 9 input and 2 output signatures match the C++ controller;
- all outputs are finite;
- 100 single-thread calls averaged about 2.71 ms per call on this server;
- the ONNX-portable actor rewrite is exactly equal to the original PyTorch
  actor on the numerical comparison used here (maximum absolute error `0`);
- IsaacLab articulation order was recovered from the packaged training data:
  joint values match the named pre-package source below `5.1e-7 rad`, and the
  selected rigid-body poses match named MuJoCo FK within `3.2e-6 m`;
- a bounded DDS test reached `DAMPING -> STAND -> LOCO -> DANCE` and ran policy
  inference without exceptions or non-finite MuJoCo states.

`model_400.pt` is only an interface smoke checkpoint, not a final quality
checkpoint.  Re-run step 1 after training; the C++ loader automatically picks
the largest exported numeric checkpoint directory.

The currently packaged default is `model_800` (checkpoint directory `800`).
Its ONNX model passes the checker and finite CPU inference validation.
